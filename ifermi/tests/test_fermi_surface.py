import unittest
from pathlib import Path

import numpy as np
from monty.serialization import loadfn
from pymatgen import Spin

from ifermi.fermi_surface import FermiSurface

test_dir = Path(__file__).resolve().parent


class FermiSurfaceTest(unittest.TestCase):
    def setUp(self):
        bs_data = loadfn(test_dir / "bs_BaFe2As2.json.gz")
        self.band_structure = bs_data["bs"]

        # for some reason BandStructure json doesn't include the structure
        self.band_structure.structure = bs_data["structure"]

        self.kpoint_dim = bs_data["dim"]
        self.ref_fs_wigner = loadfn(test_dir / "fs_BaFe2As2_wigner.json.gz")
        self.ref_fs_reciprocal = loadfn(test_dir / "fs_BaFe2As2_reciprocal.json.gz")

    def test_wigner_seitz_cell(self):
        fs = FermiSurface.from_band_structure(
            self.band_structure, self.kpoint_dim, wigner_seitz=True
        )
        self.assert_fs_equal(fs, self.ref_fs_wigner)

    def test_decimation(self):
        fs = FermiSurface.from_band_structure(self.band_structure, self.kpoint_dim)
        n_faces_orig = len(fs.isosurfaces[Spin.up][0][1])

        fs = FermiSurface.from_band_structure(
            self.band_structure, self.kpoint_dim, decimate_factor=0.8
        )
        n_faces_new = len(fs.isosurfaces[Spin.up][0][1])
        self.assertLess(n_faces_new, n_faces_orig)

    def test_reciprocal_cell(self):
        fs = FermiSurface.from_band_structure(
            self.band_structure, self.kpoint_dim, wigner_seitz=False
        )
        self.assert_fs_equal(fs, self.ref_fs_reciprocal)

    def assert_fs_equal(self, fs1: FermiSurface, fs2: FermiSurface):
        # test reciprocal space the same
        self.assertEqual(type(fs1.reciprocal_space), type(fs2.reciprocal_space))
        np.testing.assert_array_almost_equal(
            fs1.reciprocal_space.reciprocal_lattice,
            fs2.reciprocal_space.reciprocal_lattice,
            decimal=5,
        )

        # check number of spin channels in surfaces is the same
        self.assertEqual(len(fs1.isosurfaces), len(fs2.isosurfaces))

        for spin in fs1.isosurfaces.keys():
            iso1 = fs1.isosurfaces[spin]
            iso2 = fs2.isosurfaces[spin]

            # check number of spin channels in surfaces is the same
            self.assertEqual(len(iso1), len(iso2))

            for s1, s2 in zip(iso1, iso2):
                # check vertex coordinates are almost equal
                np.testing.assert_array_almost_equal(s1[0], s2[0], decimal=5)

                # check faces are exactly equal
                np.testing.assert_array_equal(s1[1], s2[1])
