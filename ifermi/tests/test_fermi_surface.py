import unittest
from os.path import join as path_join
from pkg_resources import resource_filename
import numpy as np
from numpy.testing import assert_almost_equal
import json

from pymatgen.io.vasp import Vasprun
from ifermi.fermi_surface import FermiSurface, FermiSurface2D

class IsosurfacesTestCase(unittest.TestCase):
    def setUp(self):

        structure_path = resource_filename(
            __name__,
            path_join('..', 'data', 'MgB2', 'mgb2_structure.json'))
        with open(structure_path, 'r') as f:
            self.mgb2_structure = json.load(f)

        iso_data_path = resource_filename(
            __name__,
            path_join('..', 'data', 'MgB2', 'mgb2_iso_data.json'))
        with open(iso_data_path, 'r') as f:
            self.mgb2_iso_data = json.load(f)

    def test_isosurface(self):
        """Check special points agree between saved isosurface data and newly generated isosurface data"""
        fs = FermiSurface(self.mgb2_structure)
        isosurface = fs._isosurface()

        # Test that the generated iso-surface is correct
        for i, j in isosurface:
            # self.assertIn(label, kpath_pymatgen.kpoints)
            self.assertEqual(j,
                             self.mgb2_iso_data[i])