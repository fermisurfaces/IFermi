<p align="center">
  <img alt="IFermi logo" src="https://raw.githubusercontent.com/ajsearle97/IFermi/master/docs/src/_static/logo-01.png" height="190px">
</p>

<p align="center">
    📖📖 <a href="https://ajsearle97.github.io/IFermi">Online Documentation</a> 📖📖
</p>

IFermi is a package for plotting Fermi surfaces and from *ab initio* calculation outputs. 
The main features include:

1. Plotting three-dimensional Fermi surfaces, with interactive plotting
   supported by [mayavi](https://docs.enthought.com/mayavi/mayavi/), [plotly](https://plot.ly/) and [matplotlib](https://matplotlib.org).
2. Plotting slices of three-dimensional Fermi surfaces along a specified plane.

IFermi currently only supports VASP calculations but support for additional DFT packages 
will be added in the future.

## Example

An example of the Fermi surface and a two-dimensional slice for MgB<sub>2</sub> is shown below:

![MgB2](https://raw.githubusercontent.com/ajsearle97/IFermi/master/docs/src/_static/fermi_surface_example-01.png)


## Usage

The [online documentation](https://ajsearle97.github.io/IFermi/cli.html) provides a full 
description of the available options. To summarise, three-dimensional Fermi surfaces can
be plotted from a `vasprun.xml` file using:

```bash
ifermi
```

The two-dimensional slice of a Fermi slices along the plane specified by the miller 
indices (A B C) and distance d can be plotted from a `vasprun.xml` file using:

```bash
ifermi --slice A B C d
```

### Python interface

Alternatively, IFermi can be controlled using the Python API. A full summary of the API
is given in the [API introduction page](https://ajsearle97.github.io/IFermi/plotting_using_python.html) in the documentation.

The core classes in IFermi are:

- `Inerpolator`: to take a band structure on a uniform k-point mesh and interpolate it
  onto a denser mesh.
- `FermiSurface`: to store isosurfaces and reciprocal lattice information.
- `FermiSurfacePlotter`: to plot a Fermi surface from a `FermiSurface` object.

A minimal working example to plot the Fermi surface from a `vasprun.xml` file is:

```python
from pymatgen.io.vasp.outputs import Vasprun
from ifermi.fermi_surface import FermiSurface
from ifermi.interpolator import Interpolator
from ifermi.plotter import FermiSurfacePlotter, show_plot, save_plot


vr = Vasprun("vasprun.xml")
bs = vr.get_band_structure()

# interpolate the energies to a finer k-point mesh
interpolator = Interpolator(bs)
dense_bs, kmesh = interpolator.interpolate_bands(interpolation_factor=10)
    
fs = FermiSurface.from_band_structure(dense_bs, kmesh, mu=0.0, wigner_seitz=True)
plotter = FermiSurfacePlotter(fs)
plot = plotter.get_plot(plot_type='plotly')

save_plot(plot, "fermi-surface.png")  # saves the plot to a file
show_plot(plot)  # displays an interactive plot
```

Note, if you're not running the example in a notebook then it must be proceeded by `if __name__ == '__main__':`.

## Installation

IFermi can be installed with the command:

```bash
pip install ifermi
```

IFermi is currently compatible with Python 3.5+ and relies on a number of
open-source python packages, specifically:

- [pymatgen](http://pymatgen.org) for parsing VASP calculation output.
- [BoltzTrap2](https://gitlab.com/sousaw/BoltzTraP2) for band structure interpolation.
- [matplotlib](https://matplotlib.org), [mayavi](https://docs.enthought.com/mayavi/mayavi/), and [plotly](https://plot.ly/) for three-dimensional plotting.

## What’s new?

Track changes to IFermi through the
[changelog](https://ajsearle97.github.io/IFermi/changelog.html).

## Contributing

We greatly appreciate any contributions in the form of Pull Request.
We maintain a list of all contributors [here](https://ajsearle97.github.io/IFermi/contributors.html).

## License

IFermi is made available under the MIT License.

## Acknowledgements

Alex Ganose for developing/improving code and documentation.
Sinead Griffin for suggesting the project.
