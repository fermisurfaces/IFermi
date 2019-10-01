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
           mode='rgb', interpolate_factor=8, image_format='pdf',
           dpi=400, plt=None, fonts=None, interpolate = True, mu = 0.0,
           plot_wigner_seitz = False):
    """Plot electronic band structure diagrams from vasprun.xml files.

    Args:
        filenames (:obj:`str` or :obj:`list`, optional): Path to input files.
            Vasp:
                Use vasprun.xml or vasprun.xml.gz file.
            Questaal:
                Path to a bnds.ext file. The extension will also be used to
                find site.ext and syml.ext files in the same directory.

            If no filenames are provided, sumo
            will search for vasprun.xml or vasprun.xml.gz files in folders
            named 'split-0*'. Failing that, the code will look for a vasprun in
            the current directory. If a :obj:`list` of vasprun files is
            provided, these will be combined into a single band structure.

        code (:obj:`str`, optional): Calculation type. Default is 'vasp';
            'questaal' also supported (with a reduced feature-set).
        prefix (:obj:`str`, optional): Prefix for file names.
        directory (:obj:`str`, optional): The directory in which to save files.
        vbm_cbm_marker (:obj:`bool`, optional): Plot markers to indicate the
            VBM and CBM locations.
        projection_selection (list): A list of :obj:`tuple` or :obj:`string`
            identifying which elements and orbitals to project on to the
            band structure. These can be specified by both element and
            orbital, for example, the following will project the Bi s, p
            and S p orbitals::

                [('Bi', 's'), ('Bi', 'p'), ('S', 'p')]

            If just the element is specified then all the orbitals of
            that element are combined. For example, to sum all the S
            orbitals::

                [('Bi', 's'), ('Bi', 'p'), 'S']

            You can also choose to sum particular orbitals by supplying a
            :obj:`tuple` of orbitals. For example, to sum the S s, p, and
            d orbitals into a single projection::

                [('Bi', 's'), ('Bi', 'p'), ('S', ('s', 'p', 'd'))]

            If ``mode = 'rgb'``, a maximum of 3 orbital/element
            combinations can be plotted simultaneously (one for red, green
            and blue), otherwise an unlimited number of elements/orbitals
            can be selected.
        mode (:obj:`str`, optional): Type of projected band structure to
            plot. Options are:

                "rgb"
                    The band structure line color depends on the character
                    of the band. Each element/orbital contributes either
                    red, green or blue with the corresponding line colour a
                    mixture of all three colours. This mode only supports
                    up to 3 elements/orbitals combinations. The order of
                    the ``selection`` :obj:`tuple` determines which colour
                    is used for each selection.
                "stacked"
                    The element/orbital contributions are drawn as a
                    series of stacked circles, with the colour depending on
                    the composition of the band. The size of the circles
                    can be scaled using the ``circle_size`` option.
        circle_size (:obj:`float`, optional): The area of the circles used
            when ``mode = 'stacked'``.
        cart_coords (:obj:`bool`, optional): Whether the k-points are read as
            cartesian or reciprocal coordinates. This is only required for
            Questaal output; Vasp output is less ambiguous. Defaults to
            ``False`` (fractional coordinates).
        dos_file (:obj:'str', optional): Path to vasprun.xml file from which to
            read the density of states information. If set, the density of
            states will be plotted alongside the bandstructure.
        elements (:obj:`dict`, optional): The elements and orbitals to extract
            from the projected density of states. Should be provided as a
            :obj:`dict` with the keys as the element names and corresponding
            values as a :obj:`tuple` of orbitals. For example, the following
            would extract the Bi s, px, py and d orbitals::

                {'Bi': ('s', 'px', 'py', 'd')}

            If an element is included with an empty :obj:`tuple`, all orbitals
            for that species will be extracted. If ``elements`` is not set or
            set to ``None``, all elements for all species will be extracted.
        lm_orbitals (:obj:`dict`, optional): The orbitals to decompose into
            their lm contributions (e.g. p -> px, py, pz). Should be provided
            as a :obj:`dict`, with the elements names as keys and a
            :obj:`tuple` of orbitals as the corresponding values. For example,
            the following would be used to decompose the oxygen p and d
            orbitals::

                {'O': ('p', 'd')}

        atoms (:obj:`dict`, optional): Which atomic sites to use when
            calculating the projected density of states. Should be provided as
            a :obj:`dict`, with the element names as keys and a :obj:`tuple` of
            :obj:`int` specifying the atomic indices as the corresponding
            values. The elemental projected density of states will be summed
            only over the atom indices specified. If an element is included
            with an empty :obj:`tuple`, then all sites for that element will
            be included. The indices are 0 based for each element specified in
            the POSCAR. For example, the following will calculate the density
            of states for the first 4 Sn atoms and all O atoms in the
            structure::

                {'Sn': (1, 2, 3, 4), 'O': (, )}

            If ``atoms`` is not set or set to ``None`` then all atomic sites
            for all elements will be considered.
        total_only (:obj:`bool`, optional): Only extract the total density of
            states. Defaults to ``False``.
        plot_total (:obj:`bool`, optional): Plot the total density of states.
            Defaults to ``True``.
        legend_cutoff (:obj:`float`, optional): The cut-off (in % of the
            maximum density of states within the plotting range) for an
            elemental orbital to be labelled in the legend. This prevents
            the legend from containing labels for orbitals that have very
            little contribution in the plotting range.
        gaussian (:obj:`float`, optional): Broaden the density of states using
            convolution with a gaussian function. This parameter controls the
            sigma or standard deviation of the gaussian distribution.
        height (:obj:`float`, optional): The height of the plot.
        width (:obj:`float`, optional): The width of the plot.
        ymin (:obj:`float`, optional): The minimum energy on the y-axis.
        ymax (:obj:`float`, optional): The maximum energy on the y-axis.
        style (:obj:`list` or :obj:`str`, optional): (List of) matplotlib style
            specifications, to be composed on top of Sumo base style.
        no_base_style (:obj:`bool`, optional): Prevent use of sumo base style.
            This can make alternative styles behave more predictably.
        colours (:obj:`dict`, optional): Use custom colours for specific
            element and orbital combinations. Specified as a :obj:`dict` of
            :obj:`dict` of the colours. For example::

                {
                    'Sn': {'s': 'r', 'p': 'b'},
                    'O': {'s': '#000000'}
                }

            The colour can be a hex code, series of rgb value, or any other
            format supported by matplotlib.
        yscale (:obj:`float`, optional): Scaling factor for the y-axis.
        image_format (:obj:`str`, optional): The image file format. Can be any
            format supported by matplotlib, including: png, jpg, pdf, and svg.
            Defaults to pdf.
        dpi (:obj:`int`, optional): The dots-per-inch (pixel density) for
            the image.
        plt (:obj:`matplotlib.pyplot`, optional): A
            :obj:`matplotlib.pyplot` object to use for plotting.
        fonts (:obj:`list`, optional): Fonts to use in the plot. Can be a
            a single font, specified as a :obj:`str`, or several fonts,
            specified as a :obj:`list` of :obj:`str`.

    Returns:
        If ``plt`` set then the ``plt`` object will be returned. Otherwise, the
        method will return a :obj:`list` of filenames written to disk.
    """
    # if not filenames:
    #     filenames = find_vasprun_files()
    # elif isinstance(filenames, str):
    #     filenames = [filenames]

    # # don't save if pyplot object provided
    # save_files = False if plt else True

    vr = Vasprun("/Users/amyjade/Documents/3rdYear/URAP/PlottingProgram/dataMgB2/vasprun.xml")
    bs = vr.get_band_structure()

    if not interpolate:

        interpolate_factor = 1

    interpolater = Interpolater(bs)

    new_bs, hdims, rlattvec = interpolater.interpolate_bands(interpolate_factor)

    fs = FermiSurface(new_bs, hdims, rlattvec, mu = mu, plot_wigner_seitz = plot_wigner_seitz)

    if plot_wigner_seitz:

        bz = BrillouinZone(rlattvec)
        rc = None

    else:

        bz = None
        rc = RecipCell(rlattvec)


    plotter = FSPlotter(fs, bz = bz, rc = rc)

    plotter.fs_plot_data(plot_type = 'mayavi')

    # if save_files:
    #     basename = 'band.{}'.format(image_format)
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


# def find_vasprun_files():
#     """Search for vasprun files from the current directory.
#
#     The precedence order for file locations is:
#
#       1. First search for folders named: 'split-0*'
#       2. Else, look in the current directory.
#
#     The split folder names should always be zero based, therefore easily
#     sortable.
#     """
#     folders = glob.glob('split-*')
#     folders = sorted(folders) if folders else ['.']
#
#     filenames = []
#     for fol in folders:
#         vr_file = os.path.join(fol, 'vasprun.xml')
#         vr_file_gz = os.path.join(fol, 'vasprun.xml.gz')
#
#         if os.path.exists(vr_file):
#             filenames.append(vr_file)
#         elif os.path.exists(vr_file_gz):
#             filenames.append(vr_file_gz)
#         else:
#             logging.error('ERROR: No vasprun.xml found in {}!'.format(fol))
#             sys.exit()
#
#     return filenames
#
#
# def save_data_files(bs, prefix=None, directory=None):
#     """Write the band structure data files to disk.
#
#     Args:
#         bs (`BandStructureSymmLine`): Calculated band structure.
#         prefix (`str`, optional): Prefix for data file.
#         directory (`str`, optional): Directory in which to save the data.
#
#     Returns:
#         The filename of the written data file.
#     """
#     filename = '{}_band.dat'.format(prefix) if prefix else 'band.dat'
#     directory = directory if directory else '.'
#     filename = os.path.join(directory, filename)
#
#     if bs.is_metal():
#         zero = bs.efermi
#     else:
#         zero = bs.get_vbm()['energy']
#
#     with open(filename, 'w') as f:
#         header = '#k-distance eigenvalue[eV]\n'
#         f.write(header)
#
#         # write the spin up eigenvalues
#         for band in bs.bands[Spin.up]:
#             for d, e in zip(bs.distance, band):
#                 f.write('{:.8f} {:.8f}\n'.format(d, e - zero))
#             f.write('\n')
#
#         # calculation is spin polarised, write spin down bands at end of file
#         if bs.is_spin_polarized:
#             for band in bs.bands[Spin.down]:
#                 for d, e in zip(bs.distance, band):
#                     f.write('{:.8f} {:.8f}\n'.format(d, e - zero))
#                 f.write('\n')
#     return filename
#

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
             directory=args.directory, mode=args.mode,
             interpolate_factor=args.interpolate_factor,
             image_format=args.image_format, dpi=args.dpi, fonts=args.font)


if __name__ == "__main__":
    main()
