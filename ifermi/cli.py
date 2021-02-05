# Copyright (c) Amy Searle
# Distributed under the terms of the MIT License.

"""
A script to plot Fermi surfaces
"""


import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

__author__ = "Amy Searle"
__version__ = "0.1.0"
__maintainer__ = "Amy Searle"
__email__ = "amyjadesearle@gmail.com"
__date__ = "Sept 18, 2019"


def ifermi(
    filename: Optional[Union[Path, str]] = None,
    interpolation_factor: int = 8,
    decimate_factor: Optional[float] = None,
    mu: float = 0.0,
    azimuth: float = 45.0,
    elevation: float = 35.0,
    wigner_seitz: bool = True,
    spin: Optional["Spin"] = None,
    smooth: bool = False,
    plot_type: str = "plotly",
    slice_info: Optional[Tuple[float, float, float, float]] = None,
    output_filename: Optional[str] = None,
    scale: float = 4,
):
    """Plot Fermi surfaces from a vasprun.xml file.

    Args:
        filename: Path to input vasprun file.
        interpolation_factor: The factor by which to interpolate the bands.
        output_filename: The output file name. This will prevent the plot from being
            interactive.
        decimate_factor: Scaling factor by which to reduce the number of faces.
        mu: The level above the Fermi energy at which the isosurfaces are to be plotted.
        azimuth: The azimuth of the viewpoint in degrees. i.e. the angle subtended
            by the position vector on a sphere projected on to the x-y plane.
        elevation: The zenith angle of the viewpoint in degrees, i.e. the angle
            subtended by the position vector and the z-axis.
        wigner_seitz: Controls whether the cell is the Wigner-Seitz cell or the
            reciprocal unit cell parallelepiped.
        spin: The spin channel to plot. By default plots both spin channels. Should be
            a pymatgen ``Spin`` object.
        smooth: If True, will smooth FermiSurface. Requires PyMCubes. See
            ``compute_isosurfaces`` for more information.
        plot_type: Method used for plotting. Valid options are: "matplotlib", "plotly",
            "mayavi".
        slice_info: Slice through the Brillouin zone. Given as the plane normal and
            distance form the plane in fractional coordinates: E.g., ``[1, 0, 0, 0.2]``
            where ``(1, 0, 0)`` are the miller indices and ``0.2`` is the distance from
            the Gamma point.
        scale: Scale for the figure size. Increases resolution but does not change the
            relative size of the figure and text.

    Returns:
        The filename written to disk.
    """
    from pymatgen.io.vasp.outputs import Vasprun

    from ifermi.fermi_surface import FermiSurface
    from ifermi.interpolator import Interpolator
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

    if mlab and plot_type == "mayavi" and output_filename is not None:
        # handle mlab non interactive plots
        mlab.options.offscreen = True

    if not filename:
        filename = find_vasprun_file()

    vr = Vasprun(filename)
    bs = vr.get_band_structure()

    interpolator = Interpolator(bs)
    interp_bs, kpoint_dim = interpolator.interpolate_bands(interpolation_factor)

    fs = FermiSurface.from_band_structure(
        interp_bs,
        kpoint_dim,
        mu=mu,
        wigner_seitz=wigner_seitz,
        decimate_factor=decimate_factor,
        smooth=smooth,
    )
    if slice_info:
        plane_normal = slice_info[:3]
        distance = slice_info[3]

        fermi_slice = fs.get_fermi_slice(plane_normal, distance)
        plotter = FermiSlicePlotter(fermi_slice)

        plot = plotter.get_plot(spin)
    else:
        plotter = FermiSurfacePlotter(fs)
        plot = plotter.get_plot(
            plot_type=plot_type,
            spin=spin,
            azimuth=azimuth,
            elevation=elevation,
        )

    if output_filename is None:
        show_plot(plot)
    else:
        print("Saving plot to {}".format(output_filename))
        save_plot(plot, output_filename, scale=scale)


def find_vasprun_file():
    """Search for vasprun files from the current directory.

    Will look for vasprun.xml or vasprun.xml.gz files.
    """
    for file in ["vasprun.xml", "vasprun.xml.gz"]:
        if os.path.exists(file):
            return file

    print("ERROR: No vasprun.xml found in current directory")
    sys.exit()


def _get_fs_parser():
    parser = argparse.ArgumentParser(
        description="""
    ifermi is a package for plotting Fermi-surfaces""",
        epilog="""
    Author: {}
    Version: {}
    Last updated: {}""".format(
            __author__, __version__, __date__
        ),
    )

    parser.add_argument(
        "-f", "--filename", default=None, metavar="F", help="vasprun.xml file to plot"
    )
    parser.add_argument(
        "-o", "--output", dest="output_filename", metavar="O", help="output filename"
    )
    parser.add_argument(
        "-m",
        "--mu",
        default=0.0,
        type=float,
        help="offset from the Fermi level at which to calculate Fermi surface",
    )
    parser.add_argument(
        "-i",
        "--interpolation-factor",
        type=int,
        default=8,
        dest="interpolation_factor",
        metavar="N",
        help="interpolation factor for band structure (default: 8)",
    )
    parser.add_argument(
        "-r",
        "--reciprocal-cell",
        dest="wigner_seitz",
        action="store_false",
        help="use the reciprocal lattice rather than Wigner-Seitz cell",
    )
    parser.add_argument(
        "--spin",
        type=string_to_spin,
        default=None,
        help="select spin channel (options: up, 1; down, -1)",
    )
    parser.add_argument(
        "-a",
        "--azimuth",
        type=float,
        default=45.0,
        metavar="A",
        help="viewpoint azmith angle in degrees (default: 45)",
    )
    parser.add_argument(
        "-e",
        "--elevation",
        type=float,
        default=35.0,
        metavar="E",
        help="viewpoint elevation (zenith) angle in degrees (default: 35)",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="smooth the Fermi surface",
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="plot_type",
        default="plotly",
        help="plotting type (options: matplotlib, plotly, mayavi)",
    )
    parser.add_argument(
        "--slice",
        type=float,
        nargs=4,
        metavar="N",
        help="slice through the Brillouin zone (format: j k l distance)",
    )
    parser.add_argument(
        "--decimate-factor",
        type=float,
        default=None,
        dest="decimate_factor",
        metavar="N",
        help="factor by which to decimate Fermi surfaces (i.e., 0.8 gives 20 %% fewer "
        "faces)",
    )
    parser.add_argument(
        "--scale", type=float, default=4, help="scale for image resolution (default: 4)"
    )
    return parser


def main():
    args = _get_fs_parser().parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=UnicodeWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

    ifermi(
        filename=args.filename,
        interpolation_factor=args.interpolation_factor,
        decimate_factor=args.decimate_factor,
        mu=args.mu,
        plot_type=args.plot_type,
        azimuth=args.azimuth,
        elevation=args.elevation,
        spin=args.spin,
        smooth=args.smooth,
        wigner_seitz=args.wigner_seitz,
        slice_info=args.slice,
        output_filename=args.output_filename,
        scale=args.scale,
    )


def string_to_spin(spin_string):
    """Function to convert 'spin' cli argument to pymatgen Spin object"""
    from pymatgen import Spin

    if spin_string in ["up", "Up", "1", "+1"]:
        return Spin.up

    elif spin_string in ["down", "Down", "-1"]:
        return Spin.down

    elif spin_string is None:
        return None

    else:
        raise ValueError("Unable to parse 'spin' argument")


if __name__ == "__main__":
    main()
