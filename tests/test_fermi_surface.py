import unittest
from pathlib import Path

import numpy as np
from monty.serialization import loadfn
from pymatgen.electronic_structure.core import Spin

from ifermi.surface import FermiSurface

try:
    import open3d
except ImportError:
    open3d = None

test_dir = Path(__file__).resolve().parent


class FermiSurfaceTest(unittest.TestCase):
    def setUp(self):
        bs_data = loadfn(test_dir / "bs_BaFe2As2.json.gz")
        self.band_structure = bs_data["bs"]

        # for some reason BandStructure json doesn't include the structure
        self.band_structure.structure = bs_data["structure"]

        self.ref_fs_wigner = loadfn(test_dir / "fs_BaFe2As2_wigner.json.gz")
        self.ref_fs_reciprocal = loadfn(test_dir / "fs_BaFe2As2_reciprocal.json.gz")

    def test_wigner_seitz_cell(self):
        fs = FermiSurface.from_band_structure(self.band_structure, wigner_seitz=True)
        self.assert_fs_equal(fs, self.ref_fs_wigner)

    @unittest.skipIf(open3d is None, "open3d not installed")
    def test_decimation(self):
        fs = FermiSurface.from_band_structure(self.band_structure)
        n_faces_orig = len(fs.isosurfaces[Spin.up][0].faces)

        fs = FermiSurface.from_band_structure(self.band_structure, decimate_factor=0.8)
        n_faces_new = len(fs.isosurfaces[Spin.up][0].faces)
        assert n_faces_new < n_faces_orig

    def test_reciprocal_cell(self):
        fs = FermiSurface.from_band_structure(self.band_structure, wigner_seitz=False)
        self.assert_fs_equal(fs, self.ref_fs_reciprocal)

    def assert_fs_equal(self, fs1: FermiSurface, fs2: FermiSurface):
        # test reciprocal space the same
        assert isinstance(fs1, FermiSurface)
        assert isinstance(fs2, FermiSurface)
        np.testing.assert_array_almost_equal(
            fs1.reciprocal_space.reciprocal_lattice,
            fs2.reciprocal_space.reciprocal_lattice,
            decimal=5,
        )

        # check number of spin channels in surfaces is the same
        assert len(fs1.isosurfaces) == len(fs2.isosurfaces)

        for spin in fs1.isosurfaces:
            iso1 = fs1.isosurfaces[spin]
            iso2 = fs2.isosurfaces[spin]

            # check number of spin channels in surfaces is the same
            assert len(iso1) == len(iso2)

            for s1, s2 in zip(iso1, iso2):
                # check vertex coordinates are almost equal
                # first sort vertices to make sure they are comparable
                v1 = s1.vertices
                v2 = s2.vertices
                v1 = v1[np.lexsort((v1[:, 2], v1[:, 1], v1[:, 0]))]
                v2 = v2[np.lexsort((v2[:, 2], v2[:, 1], v2[:, 0]))]

                np.testing.assert_array_almost_equal(v1, v2, decimal=5)

                # check set faces are exactly equal
                f1 = {tuple(x) for x in s1.faces}
                f2 = {tuple(x) for x in s2.faces}
                np.testing.assert_array_equal(f1, f2)
