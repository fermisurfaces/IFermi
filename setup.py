""""
IFermi: Fermi surface plotting tool from DFT output files.
"""

from setuptools import setup, find_packages


with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="ifermi",
    version="0.1.2",
    description="Fermi surface plotting tool from DFT output",
    url="https://github.com/asearle13/IFermi",
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
    test_suite="nose.collector",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "colorlover",
        "matplotlib",
        "pymatgen>=2017.12.30",
        "BoltzTraP2",
        "trimesh",
        "meshcut",
        "scikit-image",
        "monty",
        "jupyter",
        "psutil"
    ],
    extras_require={
        "mayavi": ["mayavi", "mlabtex", "vtk"],
        "plotly": ["plotly"],
        "docs": [
            "sphinx",
            "sphinx-argparse",
            "sphinx-autodoc-typehints",
            "m2r",
        ],
        "dev": ["black"],
        "tests": ["nose", "coverage", "coveralls"],
        ':python_version < "3.7"': ["dataclasses>=0.6"]
    },
    data_files=["LICENSE"],
    entry_points={"console_scripts": ["ifermi = ifermi.cli:main"]},
)
