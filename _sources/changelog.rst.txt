Change Log
==========

[Unreleased]
------------

v0.1.5
------

Enhancements:

- Simplified interpolator and FermiSurface generation api.

Bug fixes:

- Fixed bug where the Fermi surface was not exactly centered in reciprocal space.


v0.1.4
------

Enhancements:

- Standardized plots for all plotting backends.
- Added ability to change viewpoint in static plots.
- Documentation overhaul, including new contributors page.
- Added example jupyter notebook.
- API updated to separate plotting and saving files. Allows composing multiple Fermi
  surfaces.
- Surface decimation and smoothing (@mkhorton).
- Support for ``crystal_toolkit`` (@mkhorton).

Bug fixes:

- Fermi level is no longer adjusted from VASP value.
- Bug fix for smoothing (@mdforti).
- Fixed latex labex in plotly (@mdforti).
- Better support for spin polarized materials.

v0.0.4
------

Initial release.
