---
title: 'IFermi: A python library for Fermi surface generation and analysis'
tags:
  - Python
  - electronic structure
  - fermi surface
  - spin texture
  - materials science
  - chemistry
  - physics
authors:
  - name: Alex M Ganose^[equal contribution]
    orcid: 0000-0002-4486-3321
    affiliation: 1
  - name: Amy Searle^[equal contribution]
    affiliation: "2, 3, 4"
  - name: Anubhav Jain
    orcid: 0000-0001-5893-9967
    affiliation: 1
  - name: Sinéad M Griffin
    orcid: 0000-0002-9943-4866
    affiliation: "2, 3"
affiliations:
 - name: Energy Technologies Area, Lawrence Berkeley National Laboratory, Berkeley, California 94720, USA
   index: 1
 - name: Materials Science Division, Lawrence Berkeley National Laboratory, Berkeley, California 94720, USA
   index: 2
 - name: Molecular Foundry, Lawrence Berkeley National Laboratory, Berkeley, California 94720, USA
   index: 3
 - name: Clarendon Laboratory, Department of Physics, University of Oxford, OX1 3PU, UK
   index: 4
date: 1 March 2021
bibliography: paper.bib
---

# Summary

The Fermi surface is an important tool for understanding the electronic, optical, and
magnetic properties of metals and doped semiconductors [@review].
It defines the surface in reciprocal space that divides unoccupied and occupied
states at zero temperature.
The topology of the Fermi surface impacts a variety of quantum phenomena including
superconductivity, topological insulation, and ferromagnetism, and it can be used to
predict the complex behaviour of systems without requiring more detailed computations.
For example: (i) large nested Fermi sheets are a characteristic of charge density ordering [@cdw];
(ii) the size and position of Fermi pockets are indicators of high-performance
thermoelectrics [@thermoelectrics]; and (iii) the average group velocities across the
Fermi surface control the sensitivity of materials for dark matter detection [@darkmatter].
IFermi is a Python library for the generation, analysis, and visualisation of Fermi
surfaces that can facilitate sophisticated analyses of Fermi 
surface properties.

# Statement of need

Many tools already exist for the generation of Fermi surfaces from *ab initio* band 
structure calculations. For example, several electronic structure codes such as 
CASTEP [@castep] and QuantumATK [@quantumatk] include integrated tools for obtaining Fermi 
surfaces. Furthermore, 
software such as FermiSurfer [@fermisurfer], pyprocar [@pyprocar], BoltzTraP2 [@boltztrap2], 
and XCrysDen [@xcrysden] interface with common 
density functional theory packages and can plot Fermi surfaces from their 
outputs. All such packages, however, are only designed to visualise Fermi surfaces
and do not expose any application programming interfaces (APIs) for analysing and 
manipulating Fermi surfaces as objects. In IFermi, we address these limitations by developing
a fully-featured Python library for representing and processing Fermi surfaces
and Fermi slices. We also implement a novel algorithm for determining the dimensionality
of Fermi surfaces using the connectivity across periodic boundaries and the
Euler characteristic of the isosurface mesh. IFermi, therefore, enables the 
programmatic characterisation of Fermi surfaces and can be used as a foundational 
library for investigating complex Fermi surface properties such as nesting.

# IFermi

IFermi is a Python 3.6+ library and set of command-line tools for the generation, 
analysis, and visualisation of Fermi surfaces and Fermi slices. The goal of the library 
is to provide fully-featured `FermiSurface` and `FermiSlice` objects that allow for easy 
manipulation and programmatic analysis. The main features of the package include: 

- The Fourier interpolation of electronic band structures onto dense k-point meshes 
  required to obtain high resolution Fermi surfaces.
- The extraction of Fermi surfaces and Fermi slices from electronic band structures.
- A rich API for representing and manipulating Fermi surface objects.
- The projection of arbitrary properties onto Fermi surfaces and Fermi slices.
- Algorithms to calculate Fermi surface properties, including dimensionality, orientation, 
  and averaged projections such as Fermi velocities.
- Interactive visualisation of Fermi surfaces and slices, and their projections such as 
  spin-texture, with support for mayavi [@mayavi], plotly [@plotly] and matplotlib [@matplotlib]. 
  Examples of the graphics produced by IFermi are presented in Figure 1.

![Examples of Fermi surfaces and two-dimensional slices produced by IFermi. Fermi surface of MgB$_2$ with group velocity projections shown by (a) the isosurface color and (b) arrows colored by the scalar projection onto the [0 0 1] axis. (c) Spin texture of BiSb indicating Rashba splitting.](docs/src/_static/ifermi-example-01.png)
  
In addition to the Python library, IFermi includes several command-line tools that can
perform common tasks such as calculating Fermi surface dimensionality and Fermi velocities.
IFermi uses the pymatgen [@pymatgen] library for parsing first-principles calculation 
outputs and therefore supports all electronic structure codes supported therein. 
At the time of writing this 
comprises Vienna *ab initio* Simulation Package (VASP), ABINIT, and CP2K. IFermi also 
relies on  several open source packages, such as BoltzTraP2 [@boltztrap2] for Fourier 
interpolation, trimesh [@trimesh] for processing triangular meshes, and scikit-image 
[@scikitimage] for generating isosurfaces using the marching cubes algorithm developed 
by @marchingcubes.

# Author Contributions

The library and command-line tools were written by AS and AMG.
SMG designed and led the project.
The first draft of the manuscript was written by AMG with input from all co-authors. 

# Conflicts of Interest

There are no conflicts to declare.

# Acknowledgements

This work was funded by the DOE Basic Energy Sciences program -- the Materials Project 
-- under Grant No. KC23MP.
Computational resources were provided by the National Energy Research Scientific 
Computing Center and the Molecular Foundry, DoE Office of Science User Facilities 
supported by the Office of Science of the U.S. Department of Energy under Contract No. 
DE-AC02-05CH11231. The work performed at the Molecular Foundry was supported by the 
Office of Science, Office of Basic Energy Sciences, of the U.S. Department of Energy 
under the same contract. 
We would like to acknowledge additional suggestions and code contributions from
Matthew Horton and Mariano Forti.
We would like to acknowledge many fruitful discussions with Katherine Inzani.

# References
