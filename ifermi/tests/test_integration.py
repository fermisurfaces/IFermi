import unittest
from pathlib import Path
from ifermi.interpolator import Interpolater
from ifermi.plotter import FSPlotter
from ifermi.fermi_surface import FermiSurface
from pymatgen import Spin
from pymatgen.io.vasp.outputs import Vasprun

example_dir = Path("../../examples")


class IntegrationTest(unittest.TestCase):

    def setUp(self):
        vr = Vasprun(example_dir / "MgB2/vasprun.xml")
        bs = vr.get_band_structure()
        self.band_structure = bs

    def test_integration_wigner_seitz(self):
        interpolater = Interpolater(self.band_structure)
        new_bs, kpoint_dim = interpolater.interpolate_bands(1)
        fs = FermiSurface.from_band_structure(new_bs, kpoint_dim)
        plotter = FSPlotter(fs)
        plotter.plot(plot_type='plotly', interactive=False)
        plotter.plot(plot_type='mpl', interactive=False)
        plotter.plot(plot_type='mayavi', interactive=False)

    def test_integration_reciprocal(self):
        interpolater = Interpolater(self.band_structure)
        new_bs, kpoint_dim = interpolater.interpolate_bands(1)
        fs = FermiSurface.from_band_structure(new_bs, kpoint_dim, wigner_seitz=False)
        plotter = FSPlotter(fs)
        plotter.plot(plot_type='plotly', interactive=False)
        plotter.plot(plot_type='mpl', interactive=False)
        plotter.plot(plot_type='mayavi', interactive=False)

    def test_integration_spin(self):
        interpolater = Interpolater(self.band_structure)
        new_bs, kpoint_dim = interpolater.interpolate_bands(1)
        fs = FermiSurface.from_band_structure(new_bs, kpoint_dim, spin=Spin.up)
        plotter = FSPlotter(fs)
        plotter.plot(plot_type='plotly', interactive=False)
        plotter.plot(plot_type='mpl', interactive=False)
        plotter.plot(plot_type='mayavi', interactive=False)

    def tearDown(self):
        output_file = Path("fermi_surface.png")
        output_file.unlink()
