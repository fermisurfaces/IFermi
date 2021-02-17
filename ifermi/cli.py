"""Command line tools for generating and plotting Fermi surfaces."""

import os
import sys
import warnings

import click
from click import option

from ifermi.defaults import AZIMUTH, ELEVATION, SCALE, SYMPREC, VECTOR_SPACING

plot_type = click.Choice(["matplotlib", "plotly", "mayavi"], case_sensitive=False)
spin_type = click.Choice(["up", "down"], case_sensitive=False)
projection_type = click.Choice(["velocity", "spin"], case_sensitive=False)


@click.group(
    context_settings=dict(help_option_names=["-h", "--help"], max_content_width=120)
)
def cli():
    """IFermi is a tool for the generation, analysis and plotting of Fermi surfaces."""

    def _warning(message, *_, **__):
        click.echo("WARNING: {}\n".format(message))

    warnings.showwarning = _warning

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=UnicodeWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")


@cli.command()
@option("-f", "--filename", help="vasprun.xml file to plot")
@option(
    "-m",
    "--mu",
    default=0.0,
    help="offset from the Fermi level at which to calculate Fermi surface",
    show_default=True,
)
@option(
    "-i",
    "--interpolation-factor",
    default=8.0,
    help="interpolation factor for band structure",
    show_default=True,
)
@option("--projection", type=projection_type, help="projection type")
@option(
    "--projection-axis",
    nargs=3,
    type=float,
    help="use dot product of projections onto cartesian axis (e.g. 0 0 1)",
)
def info(filename, **kwargs):
    """Calculate information about the Fermi surface."""
    fs = _get_fermi_surface(
        filename=filename,
        interpolation_factor=kwargs["interpolation_factor"],
        projection=kwargs["projection"],
        mu=kwargs["mu"],
        decimate_factor=None,
        smooth=False,
        wigner_seitz=False
    )


@cli.command()
@option("-f", "--filename", help="vasprun.xml file to plot")
@option("-o", "--output", "output_filename", help="output filename")
@option(
    "-m",
    "--mu",
    default=0.0,
    help="offset from the Fermi level at which to calculate Fermi surface",
    show_default=True,
)
@option(
    "-i",
    "--interpolation-factor",
    default=8.0,
    help="interpolation factor for band structure",
    show_default=True,
)
@option(
    "--wigner/--no-wigner",
    "wigner_seitz",
    default=True,
    help="use Wigner-Seitz cell rather than reciprocal lattice parallelepiped",
    show_default=True,
)
@option(
    "-s",
    "--symprec",
    default=SYMPREC,
    help="symmetry precision in Å",
    show_default=True,
)
@option(
    "-a",
    "--azimuth",
    default=AZIMUTH,
    help="viewpoint azimuth angle in °",
    show_default=True,
)
@option(
    "-e",
    "--elevation",
    default=ELEVATION,
    help="viewpoint elevation angle in °",
    show_default=True,
)
@option(
    "-t",
    "--type",
    "plot_type",
    default="plotly",
    type=plot_type,
    help="plotting type",
    show_default=True,
)
@option("--projection", type=projection_type, help="projection type")
@option(
    "--color-projection/--no-color-projection",
    default=True,
    help="color Fermi surface projections",
    show_default=True,
)
@option("--projection-colormap", help="matplotlib colormap name for projections")
@option(
    "--vector-projection/--no-vector-projection",
    help="show vector projections as arrows",
    show_default=True,
)
@option("--vector-colormap", help="matplotlib colormap name for vectors")
@option(
    "--projection-axis",
    nargs=3,
    type=float,
    help="color projection by projecting onto cartesian axis (e.g. 0 0 1)",
)
@option(
    "--vector-spacing",
    default=VECTOR_SPACING,
    help="spacing between projection arrows",
    show_default=True,
)
@option("--cmin", type=float, help="minimum intensity on projection colorbar")
@option("--cmax", type=float, help="maximum intensity on projection colorbar")
@option("--vnorm", type=float, help="value by which to normalise vector lengths")
@option(
    "--scale-linewidth",
    is_flag=True,
    help="scale Fermi slice thickness by projection",
    show_default=True,
)
@option(
    "--hide-surface", is_flag=True, help="hide the Fermi surface", show_default=True
)
@option(
    "--hide-labels",
    is_flag=True,
    help="hide the high-symmetry k-point labels",
    show_default=True,
)
@option(
    "--hide-cell", is_flag=True, help="hide reciprocal cell boundary", show_default=True
)
@option("--spin", type=spin_type, help="select spin channel", show_default=True)
@option("--smooth", is_flag=True, help="smooth the Fermi surface", show_default=True)
@option(
    "--slice",
    nargs=4,
    type=float,
    help="slice through the Brillouin zone (format: j k l dist)",
)
@option(
    "--decimate-factor",
    type=float,
    help="factor by which to decimate surfaces (i.e. 0.8 gives 20 % fewer faces)",
)
@option("--scale", default=SCALE, help="scale for image resolution", show_default=True)
def plot(filename, **kwargs):
    """Plot a Fermi surface from a vasprun.xml file."""
    from pymatgen.electronic_structure.core import Spin
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

    fs = _get_fermi_surface(
        filename=filename,
        interpolation_factor=kwargs["interpolation_factor"],
        projection=kwargs["projection"],
        mu=kwargs["mu"],
        decimate_factor=kwargs["decimate_factor"],
        smooth=kwargs["smooth"],
        wigner_seitz=kwargs["wigner_seitz"]
    )

    spin = {"up": Spin.up, "down": Spin.down, None: None}[kwargs["spin"]]
    projection_axis = kwargs["projection_axis"] or None

    if kwargs["slice"]:
        plane_normal = kwargs["slice"][:3]
        distance = kwargs["slice"][3]

        fermi_slice = fs.get_fermi_slice(plane_normal, distance)
        plotter = FermiSlicePlotter(fermi_slice, symprec=kwargs["symprec"])
        fig = plotter.get_plot(
            spin=spin,
            color_projection=kwargs["color_projection"],
            vector_projection=kwargs["vector_projection"],
            projection_axis=projection_axis,
            vector_spacing=kwargs["vector_spacing"],
            cmin=kwargs["cmin"],
            cmax=kwargs["cmax"],
            vnorm=kwargs["vnorm"],
            hide_slice=kwargs["hide_surface"],
            hide_labels=kwargs["hide_labels"],
            hide_cell=kwargs["hide_cell"],
            scale_linewidth=kwargs["scale_linewidth"],
        )
    else:
        plotter = FermiSurfacePlotter(fs, symprec=kwargs["symprec"])
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
            hide_labels=kwargs["hide_labels"],
            hide_cell=kwargs["hide_cell"],
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


def _get_fermi_surface(
    filename, interpolation_factor, projection, mu, decimate_factor, smooth, wigner_seitz
):
    """Common helper method to get Fermi surface"""
    import numpy as np
    from pymatgen.electronic_structure.core import Spin
    from pymatgen.io.vasp.outputs import Vasprun

    from ifermi.interpolator import Interpolator
    from ifermi.kpoints import kpoints_from_bandstructure
    from ifermi.surface import FermiSurface

    if not filename:
        filename = find_vasprun_file()

    parse_projections = projection == "spin"
    vr = Vasprun(filename, parse_projected_eigen=parse_projections)
    bs = vr.get_band_structure()

    interpolator = Interpolator(bs)
    interp_bs, velocities = interpolator.interpolate_bands(
        interpolation_factor, return_velocities=True
    )

    projection_data = None
    projection_kpoints = None
    if projection == "velocity":
        projection_data = velocities
        projection_kpoints = kpoints_from_bandstructure(interp_bs)
    elif projection == "spin":
        if vr.projected_magnetisation is not None:
            # transpose so shape is (nbands, nkpoints, natoms, norbitals, 3)
            projection_data = vr.projected_magnetisation.transpose(1, 0, 2, 3, 4)

            # sum across all atoms and orbitals
            projection_data = projection_data.sum(axis=(2, 3))
            projection_data /= np.linalg.norm(projection_data, axis=-1)[..., None]

            projection_data = {Spin.up: projection_data}
            projection_kpoints = kpoints_from_bandstructure(bs)
        else:
            click.echo(
                "ERROR: Band structure does not include spin projections.\n"
                "Ensure calculation was run with LSORBIT or LNONCOLLINEAR = True "
                "and LSORBIT = 11."
            )
            sys.exit()

    return FermiSurface.from_band_structure(
        interp_bs,
        mu=mu,
        wigner_seitz=wigner_seitz,
        decimate_factor=decimate_factor,
        smooth=smooth,
        projection_data=projection_data,
        projection_kpoints=projection_kpoints,
    )

