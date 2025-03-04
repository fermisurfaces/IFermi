# Change log

## v0.3.6

* Fix plotting bug #437.

## v0.3.5

* get_cmap deprecation by @tschaume in https://github.com/fermisurfaces/IFermi/pull/410

## v0.3.4

Fix compatibility with latest spglib.

## v0.3.3

Enhancements:

* Catch Fermi surfaces and slices that don't cross any bands by @utf in https://github.com/fermisurfaces/IFermi/pull/350

## v0.3.2

The minimum supported Python version is now 3.9.

Enhancements:

- Added support for recent trimesh versions.
- Docs revamped.
- Moved to pyproject.toml instead of setup.py
- Tests moved out of the source folder.


## v0.3.1

Fixed `hide_cell` plotting option.

## v0.3.0

New features:

- Support for plotting individual bands. Specified using the `--plot-index` command
  line option. (@aj-searle)

Enhancements:

- Fixed high-symmetry points markers.
- Updated dependencies.

## v0.2.6

Bug fixes:

- Fixed serialization issues.

## v0.2.5

Bug fixes:

- Better handling of integer decimation factors.

## v0.2.4

Enhancements:

- Added function to trim band structure to within a energy cutoff.
- Improved decimation options.

## v0.2.3

Publish IFermi on zenodo.

## v0.2.2

Saving interactive html plots is now possible using the plotly backend with:
`ifermi plot --output filename.html`.

## v0.2.1

Bug fixes:

- Fixed interpolation of projections for 1D slices.
- Fixed position of high-symmetry labels.

## v0.2.0

This version completely overhauls the Python API and command-line tools. The major
changes are:

- Support for projecting properties onto surface faces and isoline segments. The
  command-line utilities include support for group velocities and spin texture.
- New tools for calculating Fermi surface dimensionality and orientation based on
  the connectivity across periodic boundary conditions.
- New tools for calculating Fermi surface properties such as area and for averaging
  projections across the Fermi surface. This enables the calculation of Fermi velocities.
- New visualisation tools for Fermi surfaces and slices with projections. Fermi surfaces
  can now be colored by the surface properties. Additionally, vector properties
  can be indicated with arrows. This allows for the visualisation of spin texture.

Command line changes:

IFermi now has a new command line interface. There are two subcommands:

- `ifermi info`: for calculating Fermi surface properties and dimensionalities.
- `ifermi plot`: for visualisation of Fermi surfaces and slices.

API additions:

- `FermiSurface` and `FermiSlice` objects now support projections.
- Added `Isosurface` and `Isoline` classes.
- Added many analysis functions to the `FermiSurface` and `FermiSlice` modules.
- New `analysis` module containing algorithms for:

  - Calculating Fermi surface dimensionality and orientation.
  - Uniformly sampling isosurfaces and isolines.
  - Determining the connectivity of isosurfaces and isolines.
  - Interpolating and smoothing isolines.

API changes:

- `fermi_surface` module renamed `surface`.
- `FermiSlice` class and related functions moved to `slice` module.
- `plotter` module renamed `plot`.
- `interpolation` module renamed `interpolate`, and `Interpolator` class
  renamed `FourierInterpolator`.

## v0.1.5

Enhancements:

- Simplified interpolator and FermiSurface generation api.

Bug fixes:

- Fixed bug where the Fermi surface was not exactly centered in reciprocal space.


## v0.1.4

Enhancements:

- Standardized plots for all plotting backends.
- Added ability to change viewpoint in static plots.
- Documentation overhaul, including new contributors page.
- Added example jupyter notebook.
- API updated to separate plotting and saving files. Allows composing multiple Fermi
  surfaces.
- Surface decimation and smoothing (@mkhorton).
- Support for `crystal_toolkit` (@mkhorton).

Bug fixes:

- Fermi level is no longer adjusted from VASP value.
- Bug fix for smoothing (@mdforti).
- Fixed latex labels in plotly (@mdforti).
- Better support for spin polarized materials.

## v0.0.4

Initial release.
