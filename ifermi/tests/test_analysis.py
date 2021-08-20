import unittest
from pathlib import Path

from monty.serialization import loadfn
from pymatgen.electronic_structure.core import Spin

from ifermi.plot import FermiSlicePlotter, FermiSurfacePlotter
from ifermi.surface import FermiSurface

test_dir = Path(__file__).resolve().parent


class FermiSurfaceTest(unittest.TestCase):
    def setUp(self):
        bs_data = loadfn(test_dir / "bs_BaFe2As2.json.gz")
        self.band_structure = bs_data["bs"]

        # for some reason BandStructure json doesn't include the structure
        self.band_structure.structure = bs_data["structure"]

        self.ref_fs_wigner = loadfn(test_dir / "fs_BaFe2As2_wigner.json.gz")
        self.ref_fs_reciprocal = loadfn(test_dir / "fs_BaFe2As2_reciprocal.json.gz")

    def test_plot_barcode(self):
        fs = FermiSurface.from_band_structure(
            self.band_structure,
            wigner_seitz=True,
        )

        from gtda.plotting import plot_diagram

        isosurface = fs.isosurfaces[Spin.down][4]
        print(isosurface.area)
        print(isosurface.barcode.shape)
        plot = plot_diagram(isosurface.barcode[0])
        plot.show()


