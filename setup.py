""""
IFermi: Fermi surface plotting tool from DFT output files.
"""

from setuptools import setup, find_packages
from os.path import join as path_join


with open('README.md', 'r') as file:
    long_description = file.read()

setup(
    name='ifermi',
    version='0.1.0',
    description='Fermi surface plotting tool from DFT output',
    url='https://github.com/asearle13/IFermi',
    author='Amy Searle',
    author_email='amyjadesearle@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering',
        'Topic :: Other/Nonlisted Topic',
        'Operating System :: OS Independent',
        ],
    keywords='fermi-surface pymatgen dft vasp band materials-science',
    test_suite='nose.collector',
    packages=find_packages(),
    install_requires=['sympy', 'numpy', 'scipy', 'matplotlib', 'pymatgen>=2017.12.30',
                      'colorlover', 'plotly', 'BoltzTraP2', 'mayavi', 'mlabtex',
                      'meshcut', 'scikit-image'],
    extras_require={'docs': ['sphinx', 'sphinx-argparse',
                             'sphinx-autodoc-typehints', 'm2r'],
                    'dev': ['tqdm', 'pybel', 'pebble', 'maggma'],
                    'tests': ['nose', 'coverage', 'coveralls']},
    data_files=['LICENSE'],
    entry_points={'console_scripts': ['ifermi = ifermi.cli:main']}
    )

