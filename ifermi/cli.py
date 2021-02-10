# Copyright (c) Amy Searle
# Distributed under the terms of the MIT License.

"""
This modules defines command line tools for generating and plotting Fermi surfaces.
"""

import os
import sys
import warnings

import click
from click import option

from ifermi.defaults import AZIMUTH, ELEVATION, SCALE, SYMPREC, VECTOR_SPACING

__author__ = "Amy Searle, Alex Ganose"
__version__ = "0.1.0"
__maintainer__ = "Amy Searle, Alex Ganose"
__email__ = "amyjadesearle@gmail.com"
__date__ = "Feb 09, 2021"

plot_type = click.Choice(["matplotlib", "plotly", "mayavi"], case_sensitive=False)
spin_type = click.Choice(["up", "down"], case_sensitive=False)
projection_type = click.Choice(["velocity"], case_sensitive=False)


@click.group(
    context_settings=dict(help_option_names=["-h", "--help"], max_content_width=120)
)
def cli():
    """
    ifermi is a tool for the generation, analysis and plotting of Fermi surfaces
    """

    def _warning(message, *_, **__):
        click.echo("WARNING: {}\n".format(message))

    warnings.showwarning = _warning

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=UnicodeWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")


@cli.command(context_settings=dict(show_default=True))
@option("-f", "--filename", help="vasprun.xml file to plot")
@option("-o", "--output", "output_filename", help="output filename")
@option(
    "-m",
    "--mu",
    default=0.0,
    help="offset from the Fermi level at which to calculate Fermi surface",
)
@option(
    "-i",
    "--interpolation-factor",
    default=8.0,
    help="interpolation factor for band structure",
)
@option(
    "--wigner/--no-wigner",
    "wigner_seitz",
    default=True,
    help="use Wigner-Seitz cell rather than reciprocal lattice parallelepiped",
)
@option("-s", "--symprec", default=SYMPREC, help="symmetry precision in Å")
@option("-a", "--azimuth", default=AZIMUTH, help="viewpoint azimuth angle in °")
@option("-e", "--elevation", default=ELEVATION, help="viewpoint elevation angle in °")
@option(
    "-t", "--type", "plot_type", default="plotly", type=plot_type, help="plotting type"
)
@option("--projection", type=projection_type, help="projection type")
@option(
    "--color-projection/--no-color-projection",
    default=True,
    help="color Fermi surface projections",
)
@option("--projection-colormap", help="matplotlib colormap name for projections")
@option(
    "--vector-projection/--no-vector-projection",
    help="show vector projections as arrows",
)
@option("--vector-colormap", help="matplotlib colormap name for vectors")
@option(
    "--projection-axis",
    nargs=3,
    type=float,
    help="color projection by projecting onto cartesian axis (e.g. 0 0 1)",
)
@option(
    "--vector-spacing", default=VECTOR_SPACING, help="spacing between projection arrows"
)
@option("--cmin", type=float, help="minimum intensity on projection colorbar")
@option("--cmax", type=float, help="maximum intensity on projection colorbar")
@option("--vnorm", type=float, help="value by which to normalise vector lengths")
@option("--hide-surface", is_flag=True, help="hide the Fermi surface")
@option("--spin", type=spin_type, help="select spin channel")
@option("--smooth", is_flag=True, help="smooth the Fermi surface")
@option(
    "--slice",
    nargs=4,
    type=float,
    help="slice through the Brillouin zone (format: j k l dist)",
)
@option(
    "--decimate-factor",
    type=float,
    help="factor by which to decimate surfaces (i.e. 0.8 gives 20 %% fewer faces)",
)
@option("--scale", default=SCALE, help="scale for image resolution")
def plot(filename, **kwargs):
    """
    Plot Fermi surfaces from a vasprun.xml file.
    """
    from pymatgen.electronic_structure.core import Spin
    from pymatgen.io.vasp.outputs import Vasprun

    from ifermi.fermi_surface import FermiSurface
    from ifermi.interpolator import Interpolator
    from ifermi.kpoints import get_kpoints_from_bandstructure
    from ifermi.plotter import (
        FermiSlicePlotter,
        FermiSurfacePlotter,
        save_plot,
        show_plot,
    )

    try:
        import mayavi.mlab as mlab
    except ImportError:
        mlab = False

    if kwargs["projection_colormap"]:
        kwargs["color_projection"] = kwargs["projection_colormap"]

    if kwargs["vector_colormap"]:
        kwargs["vector_projection"] = kwargs["vector_colormap"]

    output_filename = kwargs["output_filename"]
    if mlab and kwargs["plot_type"] == "mayavi" and output_filename is not None:
        # handle mlab non interactive plots
        mlab.options.offscreen = True

    if not filename:
        filename = find_vasprun_file()

    vr = Vasprun(filename)
    bs = vr.get_band_structure()

    interpolator = Interpolator(bs)
    interp_bs, velocities = interpolator.interpolate_bands(
        kwargs["interpolation_factor"], return_velocities=True
    )

    projection_data = None
    projection_kpoints = None
    if kwargs["projection"] == "velocity":
        projection_data = velocities
        projection_kpoints = get_kpoints_from_bandstructure(interp_bs)
    elif kwargs["projection"] is not None:
        click.echo("unrecognised projection type - valid options are: velocity")
        sys.exit()

    fs = FermiSurface.from_band_structure(
        interp_bs,
        mu=kwargs["mu"],
        wigner_seitz=kwargs["wigner_seitz"],
        decimate_factor=kwargs["decimate_factor"],
        smooth=kwargs["smooth"],
        projection_data=projection_data,
        projection_kpoints=projection_kpoints,
    )

    spin = {"up": Spin.up, "down": Spin.down, None: None}[kwargs["spin"]]

    if kwargs["slice"]:
        plane_normal = kwargs["slice"][:3]
        distance = kwargs["slice"][3]

        fermi_slice = fs.get_fermi_slice(plane_normal, distance)
        plotter = FermiSlicePlotter(fermi_slice, symprec=kwargs["symprec"])
        fig = plotter.get_plot(spin)
    else:
        plotter = FermiSurfacePlotter(fs, symprec=kwargs["symprec"])
        projection_axis = kwargs["projection_axis"] or None

        fig = plotter.get_plot(
            plot_type=kwargs["plot_type"],
            spin=spin,
            azimuth=kwargs["azimuth"],
            elevation=kwargs["elevation"],
            color_projection=kwargs["color_projection"],
            vector_projection=kwargs["vector_projection"],
            projection_axis=projection_axis,
            vector_spacing=kwargs["vector_spacing"],
            cmin=kwargs["cmin"],
            cmax=kwargs["cmax"],
            vnorm=kwargs["vnorm"],
            hide_surface=kwargs["hide_surface"],
        )

    if output_filename is None:
        show_plot(fig)
    else:
        click.echo("Saving plot to {}".format(output_filename))
        save_plot(fig, output_filename, scale=kwargs["scale"])


def find_vasprun_file():
    """Search for vasprun files from the current directory.

    Will look for vasprun.xml or vasprun.xml.gz files.
    """
    for file in ["vasprun.xml", "vasprun.xml.gz"]:
        if os.path.exists(file):
            return file

    click.echo("ERROR: No vasprun.xml found in current directory")
    sys.exit()
