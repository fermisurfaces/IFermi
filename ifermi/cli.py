# Copyright (c) Amy Searle
# Distributed under the terms of the MIT License.

"""
A script to plot Fermi surfaces

TODO:
 - A lot
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

from pymatgen import Spin
from pymatgen.io.vasp.outputs import Vasprun

from ifermi.plotter import FermiSlicePlotter

__author__ = "Amy Searle"
__version__ = "0.1.0"
__maintainer__ = "Amy Searle"
__email__ = "amyjadesearle@gmail.com"
__date__ = "Sept 18, 2019"


def fsplot(
    filename: Optional[Union[Path, str]] = None,
    interpolate_factor: int = 8,
    decimate_factor: Optional[float] = None,
    mu: float = 0.0,
    wigner_seitz: bool = True,
    spin: Optional[Spin] = None,
    plot_type: str = "plotly",
    interactive: bool = False,
    slice_info: Optional[Tuple[float, float, float, float]] = None,
    prefix: Optional[str] = None,
    directory: Optional[Union[Path, str]] = None,
    image_format: str = "png",
    dpi: float = 400,
):
    """Plot Fermi surfaces from a vasprun.xml file.

    Args:
        filename: Path to input vasprun file.
        interpolate_factor: The factor by which to interpolate the bands.
        decimate_factor: Scaling factor by which to reduce the number of faces.
        mu: The level above the Fermi energy at which the isosurfaces are to be plotted.
        wigner_seitz: Controls whether the cell is the Wigner-Seitz cell or the
            reciprocal unit cell parallelepiped.
        spin: The spin channel to plot. By default plots both spin channels.
        plot_type: Method used for plotting. Valid options are: "matplotlib", "plotly",
            "mayavi".
        interactive: Whether to enable interactive plots.
        prefix: Prefix for file names.
        slice_info: Slice through the Brillouin zone. Given as the plane normal and
            distance form the plane in fractional coordinates: E.g., ``[1, 0, 0, 0.2]``
            where ``(1, 0, 0)`` are the miller indices and ``0.2`` is the distance from
            the Gamma point.
        directory: The directory in which to save files.
        image_format: The image file format.
        dpi: The dots-per-inch (pixel density) for the image.

    Returns:
        The filename written to disk.
    """
    from ifermi.fermi_surface import FermiSurface
    from ifermi.interpolator import Interpolater
    from ifermi.plotter import FermiSurfacePlotter

    if not filename:
        filename = find_vasprun_file()

    vr = Vasprun(filename)
    bs = vr.get_band_structure()

    interpolater = Interpolater(bs)

    interp_bs, kpoint_dim = interpolater.interpolate_bands(interpolate_factor)
    fs = FermiSurface.from_band_structure(
        interp_bs, kpoint_dim, mu=mu, wigner_seitz=wigner_seitz,
        decimate_factor=decimate_factor
    )

    directory = directory if directory else "."
    prefix = "{}_".format(prefix) if prefix else ""

    if slice_info:
        plane_normal = slice_info[:3]
        distance = slice_info[3]

        fermi_slice = fs.get_fermi_slice(plane_normal, distance)
        plotter = FermiSlicePlotter(fermi_slice)

        output_filename = "{}fermi_slice.{}".format(prefix, image_format)
        output_filename = Path(directory) / output_filename
        plotter.plot(filename=output_filename, spin=spin)
    else:
        plotter = FermiSurfacePlotter(fs)

        output_filename = "{}fermi_surface.{}".format(prefix, image_format)
        output_filename = Path(directory) / output_filename
        plotter.plot(
            plot_type=plot_type,
            interactive=interactive,
            filename=output_filename,
            spin=spin,
        )


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
        "-f", "--filename", default=None, metavar="F", help="A vasprun.xml file to plot"
    )
    parser.add_argument(
        "-p", "--prefix", metavar="P", help="prefix for the files generated"
    )
    parser.add_argument(
        "-m",
        "--mu",
        default=0.0,
        type=float,
        help="offset from the Fermi level at which to calculate Fermi surface",
    )
    parser.add_argument(
        "-d", "--directory", metavar="D", help="output directory for files"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="enable interactive plots"
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
        "-t",
        "--type",
        dest="plot_type",
        default="plotly",
        help="plotting type (options: mpl, plotly, mayavi)",
    )
    parser.add_argument(
        "--slice",
        type=float,
        nargs=4,
        metavar="N",
        help="slice through the Brillouin zone (format: j k l distance)",
    )
    parser.add_argument(
        "--interpolate-factor",
        type=int,
        default=8,
        dest="interpolate_factor",
        metavar="N",
        help="interpolate factor for band structure " "projections (default: 4)",
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
        "--format",
        type=str,
        default="png",
        dest="image_format",
        metavar="FORMAT",
        help="image file format (options: pdf, svg, jpg, png)",
    )
    parser.add_argument(
        "--dpi", type=int, default=400, help="pixel density for image file"
    )
    return parser


def main():
    args = _get_fs_parser().parse_args()
    logging.basicConfig(
        filename="ifermi-fsplot.log",
        level=logging.INFO,
        filemode="w",
        format="%(message)s",
    )
    console = logging.StreamHandler()
    logging.info(" ".join(sys.argv[:]))
    logging.getLogger("").addHandler(console)

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=UnicodeWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

    fsplot(
        filename=args.filename,
        interpolate_factor=args.interpolate_factor,
        decimate_factor=args.decimate_factor,
        mu=args.mu,
        plot_type=args.plot_type,
        spin=args.spin,
        interactive=args.interactive,
        wigner_seitz=args.wigner_seitz,
        slice_info=args.slice,
        prefix=args.prefix,
        directory=args.directory,
        image_format=args.image_format,
        dpi=args.dpi,
    )


def string_to_spin(spin_string):
    """Function to convert 'spin' cli argument to pymatgen Spin object"""
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
