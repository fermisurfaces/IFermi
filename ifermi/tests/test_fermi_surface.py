import numpy as np
import unittest

from monty.serialization import loadfn

from ifermi.fermi_surface import FermiSurface
from pymatgen import Spin


class FermiSurfaceTest(unittest.TestCase):
    def setUp(self):
        bs_data = loadfn("bs_BaFe2As2.json.gz")
        self.band_structure = bs_data["bs"]

        # for some reason BandStructure json doesn't include the structure
        self.band_structure.structure = bs_data["structure"]

        self.kpoint_dim = bs_data["dim"]
        self.ref_fs_wigner = loadfn("fs_BaFe2As2_wigner.json.gz")
        self.ref_fs_reciprocal = loadfn("fs_BaFe2As2_reciprocal.json.gz")
        self.ref_fs_spin_up = loadfn("fs_BaFe2As2_spin_up.json.gz")

    def test_wigner_seitz_cell(self):
        fs = FermiSurface.from_band_structure(
            self.band_structure, self.kpoint_dim, wigner_seitz=True
        )
        self.assert_fs_equal(fs, self.ref_fs_wigner)

    def test_reciprocal_cell(self):
        fs = FermiSurface.from_band_structure(
            self.band_structure, self.kpoint_dim, wigner_seitz=False
        )
        self.assert_fs_equal(fs, self.ref_fs_reciprocal)

    def test_spin_up(self):
        fs = FermiSurface.from_band_structure(
            self.band_structure, self.kpoint_dim, spin=Spin.up
        )
        self.assert_fs_equal(fs, self.ref_fs_spin_up)

    def assert_fs_equal(self, fs1: FermiSurface, fs2: FermiSurface):
        # test reciprocal space the same
        self.assertEqual(type(fs1.reciprocal_space), type(fs2.reciprocal_space))
        np.testing.assert_array_almost_equal(
            fs1.reciprocal_space.reciprocal_lattice,
            fs2.reciprocal_space.reciprocal_lattice,
            decimal=5
        )

        # check number of surfaces is the same
        self.assertEqual(len(fs1.isosurfaces), len(fs2.isosurfaces))

        for s1, s2 in zip(fs1.isosurfaces, fs2.isosurfaces):
            # check vertex coordinates are almost equal
            np.testing.assert_array_almost_equal(s1[0], s2[0], decimal=5)

            # check faces are exactly equal
            np.testing.assert_array_equal(s1[1], s2[1])
