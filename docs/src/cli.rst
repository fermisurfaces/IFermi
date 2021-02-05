``ifermi`` program
====================

IFermi can be used on the command-line through the ``ifermi``
command. All of the options provided in the command-line are also accessible using the
`Python API <plotting_using_python.html>`_.

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

Interpolation factor
~~~~~~~~~~~~~~~~~~~~

The band structure extracted from the vasprun must be processed before the Fermi
surface can be generated. The two key issues are:

1. It only contains the irreducible portion of the Brillouin zone (since symmetry was
   used in the calculation) and therefore does not contain enough information to plot
   the Fermi surface across the full reciprocal lattice.
2. It was calculated on a relatively coarse k-point mesh and will therefore produce a
   rather jagged Fermi surface.

Both issues can be solved be interpolating the band structure onto a denser k-point
mesh. This is achieved by using `BoltzTraP2 <https://gitlab.com/sousaw/BoltzTraP2>`_
to Fourier interpolate the eigenvalues onto a denser mesh that covers the full
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

    Saving output files with the plotly backend requires plotly-orca to be installed.

Running the above command in the ``examples/MgB2`` directory produces the plot:

.. image:: _static/fs-1.jpg
    :height: 250px
    :align: center

Selecting spin channels
~~~~~~~~~~~~~~~~~~~~~~~

In the plot above, the spins are degenerate (the Hamiltonian does not differentiate
between the up and down spins). This is why the surface looks dappled, IFermi
is plotting two redundant sufaces. To stop it from doing this, we can specify that
only one spin component should be plotted using the ``--spin`` option. The default
is to plot both spins but a single spin channel can be selected through the names
"up" and "down". For example:

.. code-block:: bash

    ifermi --spin up

.. image:: _static/fs-spin-up.jpg
    :height: 250px
    :align: center

Fermi surface energy
~~~~~~~~~~~~~~~~~~~~

The energy level offset at which the Fermi surface is calculated is controlled by the ``--mu``
option. The energy level is given relative to the Fermi level of the VASP calculation and is given in eV.
By default, the Fermi surface is calculated at ``mu = 0``, i.e., at the Fermi level.

For gapped materials, ``mu`` must be selected so that it falls within the
conduction or valence bands otherwise no Fermi surface will be displayed. For
example. The following command will generate the Fermi surface at 1 eV above the Fermi
level:

.. code-block:: bash

    ifermi --mu 1

Changing the viewpoint
~~~~~~~~~~~~~~~~~~~~~~

The viewpoint (camera angle) can be changed using the ``--azimuth`` (``-a``) and
``--elevation`` (``-e``) options. This will change both the initial viewpoint
for interactive plots, and the final viewpoint for static plots. To summarize:

- The azimuth is the angle subtended by the viewpoint position vector on a sphere
  projected onto the x-y plane in degrees. The default is 45°.
- The elevation (or zenith) is the angle subtended by the viewpoint position vector and
  the z-axis. The default is 35°.

The viewpoint always directed to the center of the the Fermi surface (position [0 0 0]).
As an example, the viewpoint could be changed using:

.. code-block:: bash

    ifermi --azimuth 120 --elevation 5

.. image:: _static/fs-viewpoint.jpg
    :height: 250px
    :align: center

Reciprocal space
~~~~~~~~~~~~~~~~

By default, the Wigner–Seitz cell is used to contain to the Fermi surface. The
parallelepiped reciprocal lattice cell can be used instead by selecting the
``--reciprocal-cell`` option (``-r``). For example:

.. code-block:: bash

    ifermi --reciprocal-cell

Generating slices
~~~~~~~~~~~~~~~~~

IFermi can also generate two-dimensional slices of the Fermi surface along a specified
plane using the ``--slice`` option. Planes are defined by their miller indices (a b c)
and a distance from the plane, d. Most of the above options also apply to to Fermi slices.
However, slices are always plotted using matplotlib as the backend.

For example, a slice through the (0 0 1) plane can be generated using:

.. code-block:: bash

    ifermi --slice 0 0 1 0 --output slice.png

.. image:: _static/slice.png
    :height: 250px
    :align: center

Command-line interface
----------------------

.. argparse::
   :module: ifermi.cli
   :func: _get_fs_parser
   :prog: ifermi
