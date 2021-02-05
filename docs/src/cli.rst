``ifermi`` program
====================

IFermi can be used on the command-line through the ``ifermi``
command. All of the options provided in the command-line are also accessible using the
`Python API <plotting_using_python>`_.

IFermi works in 4 stages:

1. It loads a band structure from DFT calculation outputs.
2. It interpolates the band structure onto a dense k-point mesh using Fourier
   interpolation as implemented in `BoltzTraP2 <https://gitlab.com/sousaw/BoltzTraP2>`_.
3. It extracts the Fermi surface at a given energy level.
4. It plots the surface using a number of plotting backends.

.. NOTE::

    Currently, IFermi only works with VASP calculation outputs. Support for additional
    DFT packages will be added in a future release.

Usage
-----

IFermi is controlled on the command-line using the ``ifermi`` command. The only input
required is a vasprun.xml file. To plot an interactive Fermi surface just run:

.. code-block:: bash

    ifermi

IFermi will look for a vasprun.xml or vasprun.xml.gz file in the current directory.
To specify a particular vasprun file the ``--filename`` (or ``-f`` for short) option
can be used:

.. code-block:: bash

    ifermi --filename my_vasprun.xml

Plotting backend
~~~~~~~~~~~~~~~~

IFermi supports multiple plotting backends. The default is to the
`plotly <http://plotly.com>`_ package but `matplotlib <http://matplotlib.org>`_ and
`mayavi <https://docs.enthought.com/mayavi/mayavi/>`_ are also supported.

.. NOTE::

    The mayavi dependencies are not installed by default. To use this backend, follow
    the installation instructions `here <https://docs.enthought.com/mayavi/mayavi/installation.html>`_
    and then install IFermi using ``pip install ifermi[mayavi]``.

Different plotting packages can be specified using the ``--type`` option (``-t``). For
example, to use matplotlib:

.. code-block:: bash

    ifermi --type matplotlib

Output files
~~~~~~~~~~~~

By default, IFermi generates interactive plots. To generate static images, an output
file can be specified using the ``--output`` (``-o``) option. For example:

.. code-block:: bash

    ifermi --output fermi-surface.jpg

.. NOTE::

    Saving outpuut files with the plotly backend requires plotly-orca to be installed


Interpolation factor
~~~~~~~~~~~~~~~~~~~~

The band band structure extracted from the vasprun must be processed before the Fermi
surface can be generated. The two key issues are:

1. It only contains the irreducible portion of the Brillouin zone (since symmetry was
   used in the calculation) and therefore does not contain enough information to plot
   the Fermi surface across the full reciprocal lattice.
2. It was calculated on a relatively coarse k-point mesh and therefore will produce a
   rather jagged Fermi surface.

Both issues can be solved be interpolating the band structure onto a denser k-point
mesh. This is achieved by using `BoltzTraP2 <https://gitlab.com/sousaw/BoltzTraP2>`_
to Fourier interpolate the eigenvalues on to a denser mesh that covers the full
Brillouin zone.

The degree of interpolation is controlled by the ``--interpolation-factor`` (``-i``)
argument. A value of 8 (the default value), roughly indicates that the interpolated band
structure will contain 8x as many k-points. Increasing the interpolation factor will
result in more smooth Fermi surfaces. For example:

.. code-block:: bash

    ifermi --interpolation-factor 10


.. WARNING::

    As the interpolation increases, the generation of the Fermi surface and plot will
    take a longer time and can result in large file sizes.

Fermi surface energy
~~~~~~~~~~~~~~~~~~~~


Reciprocal space
~~~~~~~~~~~~~~~~

Selecting spin channels
~~~~~~~~~~~~~~~~~~~~~~~

Generating slices
~~~~~~~~~~~~~~~~~


Command-line interface
----------------------

.. argparse::
   :module: ifermi.cli
   :func: _get_fs_parser
   :prog: ifermi
