import unittest
from pathlib import Path

from monty.serialization import loadfn
from pymatgen.electronic_structure.core import Spin

from ifermi.surface import FermiSurface
from ifermi.plot import FermiSurfacePlotter, FermiSlicePlotter

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

    def test_plot_surface(self):
        fs = FermiSurface.from_band_structure(self.band_structure, wigner_seitz=True)
        plotter = FermiSurfacePlotter(fs)

        plot = plotter.get_plot(plot_type="plotly")
        plot.show()

        plot = plotter.get_plot(spin=Spin.up)
        plot.show()

        # Two following two plots should look the same

        plot = plotter.get_plot(plot_index=[1, 3])
        plot.show()

        plot = plotter.get_plot(plot_index={Spin.up: [1, 3], Spin.down: [1,3]})
        plot.show()

    def test_plot_slice(self):
        fs = FermiSurface.from_band_structure(self.band_structure, wigner_seitz=True, )
        fermi_slice = fs.get_fermi_slice(plane_normal=(0, 0, 1), distance=0)
        slice_plotter = FermiSlicePlotter(fermi_slice)

        plot = slice_plotter.get_plot()
        plot.show()

