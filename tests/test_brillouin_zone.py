import unittest
from pathlib import Path
from typing import Union

import numpy as np
from monty.serialization import loadfn

from ifermi.brillouin_zone import ReciprocalCell, WignerSeitzCell

test_dir = Path(__file__).resolve().parent


class BrillouinZoneTest(unittest.TestCase):
    def setUp(self):
        self.structure = loadfn(test_dir / "structure.json.gz")
        self.ref_rs_wigner = loadfn(test_dir / "rs_wigner.json.gz")
        self.ref_rs_reciprocal = loadfn(test_dir / "rs_reciprocal.json.gz")

    def test_wigner_seitz_cell(self):
        # test from structure
        rs = WignerSeitzCell.from_structure(self.structure)
        self.assert_rs_equal(rs, self.ref_rs_wigner)

    def test_reciprocal_cell(self):
        # test from structure
        rs = ReciprocalCell.from_structure(self.structure)
        self.assert_rs_equal(rs, self.ref_rs_reciprocal)

    def assert_rs_equal(
        self,
        rs1: Union[ReciprocalCell, WignerSeitzCell],
        rs2: Union[ReciprocalCell, WignerSeitzCell],
    ):
        # test reciprocal space the same
        self.assertEqual(type(rs1), type(rs2))
        np.testing.assert_array_almost_equal(
            rs1.reciprocal_lattice, rs2.reciprocal_lattice, decimal=5
        )

        self.assertEqual(len(rs1.faces), len(rs2.faces))
        for face1, face2 in zip(rs1.faces, rs2.faces):
            np.testing.assert_array_almost_equal(face1, face2, decimal=5)
