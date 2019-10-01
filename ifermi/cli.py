# Copyright (c) Amy Searle
# Distributed under the terms of the MIT License.

"""
A script to plot Fermi surfaces

TODO:
 - A lot
"""

from __future__ import unicode_literals
from pkg_resources import Requirement, resource_filename

import os
import sys
import glob
import logging
import argparse
import warnings

from interpolator import Interpolater
from brillouin_zone import BrillouinZone, RecipCell
from fermi_surface import FermiSurface
from plotter import *

from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

__author__ = "Amy Searle"
__version__ = "0.1.0"
__maintainer__ = "Amy Searle"
__email__ = "amyjadesearle@gmail.com"
__date__ = "Sept 18, 2019"


def fsplot(filenames=None, code='vasp', prefix=None, directory=None,
           interpolate_factor=8, mu =0., image_format='pdf',
           dpi=400, plt=None, fonts=None):
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

    fs = FermiSurface(new_bs, hdims, mu = mu, plot_wigner_seitz = plot_wigner_seitz)

    if plot_wigner_seitz:

        bz = BrillouinZone(rlattvec)
        rc = None

    else:

        bz = None
        rc = RecipCell(rlattvec)


    plotter = FSPlotter(fs, bz, rc)

    plotter.fs_plot_data()

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


def fsplot2d(filenames=None, code='vasp', prefix=None, directory=None,
           interpolate_factor=8, mu = 0., image_format='pdf',
           dpi=400, plt=None, fonts=None, plane_orig = (0., 0., 0.5),
            plane_norm = (0, 0, 1)):
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

    fs = FermiSurface(new_bs, hdims, rlattvec, mu = mu, plot_wigner_seitz = False)

    if plot_wigner_seitz:

        bz = BrillouinZone(rlattvec)
        rc = None

    else:

        bz = None
        rc = RecipCell(rlattvec)

    plotter = FSPlotter2D(fs, plane_orig = plane_orig, plane_norm = plane_norm)

    plotter.fs2d_plot_data(plot_type = 'mpl')

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


def find_vasprun_files():
    """Search for vasprun files from the current directory.

    The precedence order for file locations is:

      1. First search for folders named: 'split-0*'
      2. Else, look in the current directory.

    The split folder names should always be zero based, therefore easily
    sortable.
    """
    folders = glob.glob('split-*')
    folders = sorted(folders) if folders else ['.']

    filenames = []
    for fol in folders:
        vr_file = os.path.join(fol, 'vasprun.xml')
        vr_file_gz = os.path.join(fol, 'vasprun.xml.gz')

        if os.path.exists(vr_file):
            filenames.append(vr_file)
        elif os.path.exists(vr_file_gz):
            filenames.append(vr_file_gz)
        else:
            logging.error('ERROR: No vasprun.xml found in {}!'.format(fol))
            sys.exit()

    return filenames


def save_data_files(bs, prefix=None, directory=None):
    """Write the band structure data files to disk.

    Args:
        bs (`BandStructureSymmLine`): Calculated band structure.
        prefix (`str`, optional): Prefix for data file.
        directory (`str`, optional): Directory in which to save the data.

    Returns:
        The filename of the written data file.
    """
    filename = '{}_band.dat'.format(prefix) if prefix else 'band.dat'
    directory = directory if directory else '.'
    filename = os.path.join(directory, filename)

    if bs.is_metal():
        zero = bs.efermi
    else:
        zero = bs.get_vbm()['energy']

    with open(filename, 'w') as f:
        header = '#k-distance eigenvalue[eV]\n'
        f.write(header)

        # write the spin up eigenvalues
        for band in bs.bands[Spin.up]:
            for d, e in zip(bs.distance, band):
                f.write('{:.8f} {:.8f}\n'.format(d, e - zero))
            f.write('\n')

        # calculation is spin polarised, write spin down bands at end of file
        if bs.is_spin_polarized:
            for band in bs.bands[Spin.down]:
                for d, e in zip(bs.distance, band):
                    f.write('{:.8f} {:.8f}\n'.format(d, e - zero))
                f.write('\n')
    return filename


def _get_parser():
    parser = argparse.ArgumentParser(description="""
    ifermi is a package for plotting Fermi-surfaces""",
                                     epilog="""
    Author: {}
    Version: {}
    Last updated: {}""".format(__author__, __version__, __date__))

    parser.add_argument('-f', '--filenames', default=None, nargs='+',
                        metavar='F',
                        help="one or more vasprun.xml files to plot")
    parser.add_argument('-c', '--code', default='vasp',
                        help='Electronic structure code (default: vasp).'
                             '"questaal" also supported.')
    parser.add_argument('-p', '--prefix', metavar='P',
                        help='prefix for the files generated')
    parser.add_argument('-d', '--directory', metavar='D',
                        help='output directory for files')
    parser.add_argument('--mode', default='rgb', type=str,
                        help=('mode for orbital projections (options: rgb, '
                              'stacked)'))
    parser.add_argument('--interpolate-factor', type=int, default=4,
                        dest='interpolate_factor', metavar='N',
                        help=('interpolate factor for band structure '
                              'projections (default: 4)'))
    parser.add_argument('--style', type=str, nargs='+', default=None,
                        help='matplotlib style specifications')
    parser.add_argument('--format', type=str, default='pdf',
                        dest='image_format', metavar='FORMAT',
                        help='image file format (options: pdf, svg, jpg, png)')
    parser.add_argument('--dpi', type=int, default=400,
                        help='pixel density for image file')
    parser.add_argument('--font', default=None, help='font to use')
    return parser


def main():
    args = _get_parser().parse_args()
    logging.basicConfig(filename='ifermi-fsplot.log', level=logging.INFO,
                        filemode='w', format='%(message)s')
    console = logging.StreamHandler()
    logging.info(" ".join(sys.argv[:]))
    logging.getLogger('').addHandler(console)

    warnings.filterwarnings("ignore", category=UserWarning,
                            module="matplotlib")
    warnings.filterwarnings("ignore", category=UnicodeWarning,
                            module="matplotlib")
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="pymatgen")

    fsplot(filenames=args.filenames, code=args.code, prefix=args.prefix,
             directory=args.directory,
             interpolate_factor=args.interpolate_factor,
             image_format=args.image_format, dpi=args.dpi, fonts=args.font)


if __name__ == "__main__":
    main()

