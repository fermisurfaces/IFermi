``ifermi`` program
====================

IFermi can be used on the command-line through the ``ifermi``
command. Most of the options provided in the Python API are also accessible
on the command-line. This page details the basic usage of the program.


.. contents:: Table of Contents
   :local:
   :backlinks: None

Usage
-----

The full range of options supported by ``esea`` are detailed in the
`Command-line interface`_ section, and be can be listed using::

    ifermi -h

Basic options
~~~~~~~~~~~~~

Constructing a plot generally follows two steps. First, the energy mesh must be 
created from a DFT output file and interpolated onto a finer mesh. Next, the  
plot is invoked on this mesh.

Input options
~~~~~~~~~~~~~~~~~
The most direct way to use Esea is by running the program from a folder containing 
the DFT output files. For  VASP, for example, this must include a vasprun.xml file
and POSCAR. Since the mesh is created through Pymatgen_, their documentation will
give an in depth description of what is needed to generate an electronic strucutre
object.  

One can also manually enter an energy mesh, or read it in from a file. The formatting 
for this file has been chosen to match that of the Xcrysden _ file for convenience, 
and should be named el_mesh.dat 

Once the mesh has been interpolated, the plotting is  perfomred. For example, if one
wishes to plot the Fermi surface on a mesh which is first interpolated by a factor of 
10, the command is

		esea-fs --interpolate 10

or, to plot the three dimensional dispersion relation from -3eV below the Fermi
energy to 3eV above the Fermi energy, after interpolatning by a factor of 10, 
one should type on the command line

		esea-disp --interpolate 10 --ymin -3 --ymax 3 

If one is interested in obtaining an interpolated mesh and does not want to 
do any plotting, then the relevent command is

		esea --interpolate 10 

Describer options
~~~~~~~~~~~~~~~~~

The ``ifermi`` program is a plotting tool, and so the primary options avaliable
involve style specifications. These may be compactly specified in a style file,
which much be pointed to on invoking the plotting command with the ``--style path/to/stylefile``.
By default, labels are included- both symmetry and axis labels. These can be 
disabled using the ``--no-symmetry--labels``and ``--no-axis--labels`` options, respectively.

By default, up and down spins are treated as degenerate and one band is plotted for both 
up and down spins for a cleaner plot. If the DFT calculatin was non-collinear or a 
magnetic calcualtion was done, then the command ``--spin-on should be included.  

Finally, a text file containing the k-points of the relevent iso-surface can be generated
using the option ``-f /path/to/destination`` where /path/to/destination specified where 
the contour.dat file should be created.

Command-line interface
----------------------

.. argparse::
   :module: ifermi.cli
   :func: _get_parser
   :prog: ifermi
