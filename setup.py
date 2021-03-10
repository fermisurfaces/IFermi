""""
IFermi: Fermi surface plotting tool from DFT output files.
"""
from setuptools import find_packages, setup

from ifermi import __version__

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="ifermi",
    version=__version__,
    description="Fermi surface plotting tool from DFT output",
    url="https://github.com/fermisurfaces/IFermi",
    author="Amy Searle",
    author_email="amyjadesearle@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering",
        "Topic :: Other/Nonlisted Topic",
        "Operating System :: OS Independent",
    ],
    keywords="fermi-surface pymatgen dft vasp band materials-science",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pymatgen>=2017.12.30",
        "BoltzTraP2",
        "trimesh",
        "meshcut",
        "scikit-image",
        "monty",
        "spglib",
        "plotly",
        "pyfftw",
        "psutil",
        "click",
        "networkx",
        "tabulate",
    ],
    extras_require={
        "mayavi": ["mayavi", "mlabtex", "vtk"],
        "crystal-toolkit": ["crystal-toolkit"],
        "plotly-static": ["kaleido"],
        "decimation": ["open3d"],
        "smooth": ["PyMCubes"],
        "docs": [
            "sphinx==3.2.1",
            "sphinx-click==2.5.0",
            "sphinx_rtd_theme==0.5.0",
            "sphinx-autodoc-typehints==1.11.1",
            "m2r2==0.2.5",
            "nbsphinx",
            "nbsphinx-link",
            "ipython",
        ],
        "dev": ["black"],
        "tests": ["pytest"],
        ':python_version < "3.7"': ["dataclasses>=0.6"],
    },
    data_files=["LICENSE"],
    entry_points={"console_scripts": ["ifermi = ifermi.cli:cli"]},
)
