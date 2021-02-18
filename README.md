<img alt="IFermi logo" src="https://raw.githubusercontent.com/fermisurfaces/IFermi/master/docs/src/_static/logo-01.png" height="150px">

--------
[📖 **Online Documentation** 📖](https://fermisurfaces.github.io/IFermi)
 

IFermi is a Python (3.6+) library and command-line tools the generation, 
analysis, and visualisation of Fermi surfaces and Fermi slices. The goal of the library 
is to provide full featured FermiSurface and FermiSlice objects which allow for easy 
manipulation and analysis. The main features include:

- Interpolation of electronic band structures onto dense k-point meshes.
- Extraction of FermiSurfaces and FermiSlices from electronic band structures.
- Projection of arbitrary properties on to Fermi surfaces and Fermi slices.
- Tools to calculate Fermi surface dimensionality, orientation, and averaged projections.
- Interactive visualisation of Fermi surfaces and slices, with support for
  [mayavi](https://docs.enthought.com/mayavi/mayavi/), [plotly](https://plot.ly/) and 
  [matplotlib](https://matplotlib.org).
- Generation and visualisation of spin-texture.

IFermi currently only works with VASP calculations but support for additional DFT packages 
will be added in the future.

![MgB2](https://raw.githubusercontent.com/fermisurfaces/IFermi/master/docs/src/_static/fermi_surface_example-01.png)

## Quick start

The [online documentation](https://fermisurfaces.github.io/IFermi/cli.html) provides a full 
description of the available options. 

### Analysis

Fermi surface properties, including dimensionality and orientation can be extracted 
from a vasprun.xml file using.

```bash
ifermi info
```

```
Fermi Surface Summary
=====================

  # surfaces: 10
  Area: 32.745 Å⁻²

Isosurfaces:
~~~~~~~~~~~~

    Band    Area [Å⁻²]   Dimensionality    Orientation
  ------  ------------  ----------------  -------------
       6         1.944         2D           (0, 0, 1)
       7         4.370         1D           (0, 0, 1)
       7         2.961         2D           (0, 0, 1)
       8         3.549         1D           (0, 0, 1)
       8         3.549         1D           (0, 0, 1)
```

### Visualisation

Three-dimensional Fermi surfaces can be visualized from a `vasprun.xml` file using:

```bash
ifermi plot
```

The two-dimensional slice of a Fermi surface along the plane specified by the miller 
indices (j k l) and distance d can be plotted from a `vasprun.xml` file using:

```bash
ifermi plot --slice j k l d
```

### Python library

The `ifermi` command line tools are build on the IFermi Python library. Here is an
example of how to load DFT calculation outputs, interpolate the energies onto a dense mesh, 
generate a Fermi surface, calculate Fermi surface properties, and visualise the surface.
A more complete summary of the API is given in the [API introduction page](https://fermisurfaces.github.io/IFermi/plotting_using_python.html)
and in the [API Reference page](https://fermisurfaces.github.io/IFermi/ifermi.html) in the documentation.

```python
from pymatgen.io.vasp.outputs import Vasprun
import ifermi

# load VASP calculation outputs
vr = Vasprun("vasprun.xml")
bs = vr.get_band_structure()

# interpolate the energies onto a dense k-point mesh
interpolator = ifermi.interpolate.Interpolator(bs)
dense_bs = interpolator.interpolate_bands()

# generate the Fermi surface and calculate the dimensionality
fs = ifermi.surface.FermiSurface.from_band_structure(
    dense_bs, mu=0.0, wigner_seitz=True, calculate_dimensionality=True
)

# number of isosurfaces in the Fermi surface
fs.n_surfaces

# number of isosurfaces for each Spin channel
fs.n_surfaces_per_spin

# the total area of the Fermi surface
fs.area

# the area of each isosurface
fs.area_surfaces

# loop over all isosurfaces and check their properties
# the isosurfaces are given as a list for each spin channel
for spin, isosurfaces in fs.isosurfaces.items():
    for isosurface in isosurfaces:
        
        # the dimensionality (does the surface cross periodic boundaries)
        isosurface.dimensionality
        
        # what is the orientation
        isosurface.orientation
        
        # does the surface have face properties
        isosurface.has_properties
        
        # calculate the norms of the properties
        isosurface.properties_norms
        
        # calculate scalar projection of properties on to [0 0 1] vector
        isosurface.scalar_projection((0, 0, 1))
        
        # uniformly sample the surface faces to a consistent density
        isosurface.sample_uniform(0.1)

# plot the Fermi surface
fs_plotter = ifermi.plot.FermiSurfacePlotter(fs)
plot = fs_plotter.get_plot()

# generate Fermi slice along the (0 0 1) plane going through the Γ-point.
slice = fs.get_fermi_slice((0, 0, 1)

# number of isolines in the slice
slice.n_lines

# do the lines have segment properties
slice.has_properties

# plot slice
slice_plotter = ifermi.plot.FermiSurfacePlotter(slice)
plot = slice_plotter.get_plot()

ifermi.plot.save_plot(plot, "fermi-surface.png")  # saves the plot to a file
ifermi.plot.show_plot(plot)  # displays an interactive plot
```

## Installation

IFermi can be installed with the command:

```bash
pip install ifermi
```

IFermi is currently compatible with Python 3.6+ and relies on a number of
open-source python packages, specifically:

- [pymatgen](http://pymatgen.org) for parsing DFT calculation output.
- [BoltzTrap2](https://gitlab.com/sousaw/BoltzTraP2) for band structure interpolation.
- [trimesh](https://trimsh.org/) for manipulating isosurfaces.
- [matplotlib](https://matplotlib.org), [mayavi](https://docs.enthought.com/mayavi/mayavi/), and [plotly](https://plot.ly/) for three-dimensional plotting.

## What’s new?

Track changes to IFermi through the
[changelog](https://fermisurfaces.github.io/IFermi/changelog.html).

## Contributing

We greatly appreciate any contributions in the form of Pull Request.
We maintain a list of all contributors [here](https://fermisurfaces.github.io/IFermi/contributors.html).

## License

IFermi is made available under the MIT License.

## Acknowledgements

Developed by Amy Searle and Alex Ganose.
Sinead Griffin designed and led the project.