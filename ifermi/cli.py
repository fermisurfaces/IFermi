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
from typing import Optional, Union

from pymatgen import Spin
from pymatgen.io.vasp.outputs import Vasprun

__author__ = "Amy Searle"
__version__ = "0.1.0"
__maintainer__ = "Amy Searle"
__email__ = "amyjadesearle@gmail.com"
__date__ = "Sept 18, 2019"


def fsplot(
    filename: Optional[Union[Path, str]] = None,
    interpolate_factor: int = 8,
    mu: float = 0.0,
    wigner_seitz: bool = True,
    spin: Optional[Spin] = None,
    plot_type: str = "plotly",
    interactive: bool = False,
    prefix: Optional[str] = None,
    directory: Optional[Union[Path, str]] = None,
    image_format: str = "png",
    dpi: float = 400,
):
    """Plot Fermi surfaces from a vasprun.xml file.

    Args:
        filename: Path to input vasprun file.
        interpolate_factor: The factor by which to interpolate the bands.
        mu: The level above the Fermi energy at which the isosurfaces are to be plotted.
        wigner_seitz: Controls whether the cell is the Wigner-Seitz cell or the
            reciprocal unit cell parallelepiped.
        spin: The spin channel to plot. By default plots both spin channels.
        plot_type: Method used for plotting. Valid options are: "matplotlib", "plotly",
            "mayavi".
        interactive: Whether to enable interactive plots.
        prefix: Prefix for file names.
        directory: The directory in which to save files.
        image_format: The image file format.
        dpi: The dots-per-inch (pixel density) for the image.

    Returns:
        The filename written to disk.
    """
    from ifermi.fermi_surface import FermiSurface
    from ifermi.interpolator import Interpolater
    from ifermi.plotter import FSPlotter

    if not filename:
        filename = find_vasprun_file()

    vr = Vasprun(filename)
    bs = vr.get_band_structure()

    interpolater = Interpolater(bs)

    interp_bs, kpoint_dim = interpolater.interpolate_bands(interpolate_factor)
    fs = FermiSurface.from_band_structure(
        interp_bs, kpoint_dim, mu=mu, wigner_seitz=wigner_seitz
    )

    plotter = FSPlotter(fs)

    directory = directory if directory else "."
    prefix = "{}_".format(prefix) if prefix else ""
    output_filename = "{}fermi_surface.{}".format(prefix, image_format)
    output_filename = Path(directory) / output_filename
    plotter.plot(
        plot_type=plot_type,
        interactive=interactive,
        filename=output_filename,
        spin=spin
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
        default=0.,
        type=float,
        help="offset from the Fermi level at which to calculate Fermi surface"
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
        help="use the reciprocal lattice rather than Wigner-Seitz cell"
    )
    parser.add_argument(
        '--spin',
        type=string_to_spin,
        default=None,
        help='select spin channel (options: up, 1; down, -1)'
    )
    parser.add_argument(
        '-t',
        "--type",
        dest="plot_type",
        default="plotly",
        help="plotting type (options: mpl, plotly, mayavi)",
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
        mu=args.mu,
        plot_type=args.plot_type,
        spin=args.spin,
        interactive=args.interactive,
        wigner_seitz=args.wigner_seitz,
        prefix=args.prefix,
        directory=args.directory,
        image_format=args.image_format,
        dpi=args.dpi,
    )


def fsplot2d(
        filenames=None,
        code="vasp",
        prefix=None,
        directory=None,
        interpolate_factor=8,
        mu=0.0,
        image_format="pdf",
        dpi=400,
        plt=None,
        fonts=None,
        plane_orig=(0.0, 0.0, 0.5),
        plane_norm=(0, 0, 1),
):
    """Plot electronic band structure diagrams from vasprun.xml files.

    Args:
        filenames (:obj:`str` or :obj:`list`, optional): Path to input files.
            Vasp:
                Use vasprun.xml or vasprun.xml.gz file.

            If no filenames are provided, ifermi
            will search for vasprun.xml or vasprun.xml.gz files in folders
            named 'split-0*'. Failing that, the code will look for a vasprun in
            the current directory.

        code (:obj:`str`, optional): Calculation type. Default is 'vasp'
        prefix (:obj:`str`, optional): Prefix for file names.
        directory (:obj:`str`, optional): The directory in which to save files.
        interpolate_factor (float, optional): The factor by which to interpolate the bands
        mu (float, optional): The level above the Fermi energy at which the isosurfaces are to be
            plotted.
        image_format (:obj:`str`, optional): The image file format.
        dpi (:obj:`int`, optional): The dots-per-inch (pixel density) for
            the image.
        plt (:obj:`matplotlib.pyplot`, optional): A
            :obj:`matplotlib.pyplot` object to use for plotting.

    Returns:
        If ``plt`` set then the ``plt`` object will be returned. Otherwise, the
        method will return a :obj:`list` of filenames written to disk.
    """
    from ifermi.fermi_surface import FermiSurface
    from ifermi.interpolator import Interpolater
    from ifermi.plotter import FSPlotter, FSPlotter2D

    if not filenames:
        filenames = find_vasprun_files()
    elif isinstance(filenames, str):
        filenames = [filenames]

    # # don't save if pyplot object provided
    # save_files = False if plt else True

    vr = Vasprun(filenames)
    bs = vr.get_band_structure()

    if not interpolate:
        interpolate_factor = 1

    interpolater = Interpolater(bs)

    new_bs, hdims, rlattvec = interpolater.interpolate_bands(interpolate_factor)

    fs = FermiSurface(new_bs, hdims, rlattvec, mu=mu, plot_wigner_seitz=False)

    if plot_wigner_seitz:

        bz = WignerSeitzCell(rlattvec)
        rc = None

    else:

        bz = None
        rc = RecipCell(rlattvec)

    plotter = FSPlotter2D(fs, plane_orig=plane_orig, plane_norm=plane_norm)

    plotter.fs2d_plot_data(plot_type="mpl")

    # if save_files:
    #     basename = 'fs.{}'.format(image_format)
    #     filename = '{}_{}'.format(prefix, basename) if prefix else basename
    #     if directory:
    #         filename = os.path.join(directory, filename)
    #     plt.savefig(filename, format=image_format, dpi=dpi,
    #                 bbox_inches='tight')
    #
    #     written = [filename]
    #     written += save_data_files(bs, prefix=prefix,
    #                                directory=directory)
    #     return written
    #
    # else:
    #     return plt


def string_to_spin(spin_string):
    """Function to convert 'spin' cli argument to pymatgen Spin object"""
    if spin_string in ['up', 'Up', '1', '+1']:
        return Spin.up

    elif spin_string in ['down', 'Down', '-1']:
        return Spin.down

    elif spin_string is None:
        return None

    else:
        raise ValueError("Unable to parse 'spin' argument")


if __name__ == "__main__":
    main()
