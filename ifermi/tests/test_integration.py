import unittest
from pathlib import Path

from pymatgen import Spin
from pymatgen.io.vasp.outputs import Vasprun

from ifermi.fermi_surface import FermiSurface
from ifermi.interpolator import Interpolator
from ifermi.plotter import FermiSurfacePlotter, save_plot

test_dir = Path(__file__).resolve().parent
root_dir = test_dir / "../.."
example_dir = root_dir / "examples"


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        vr = Vasprun(example_dir / "MgB2/vasprun.xml")
        bs = vr.get_band_structure()
        self.band_structure = bs
        self.output_file = test_dir / "fs.png"

    def test_integration_wigner_seitz(self):
        interpolator = Interpolator(self.band_structure)
        new_bs, kpoint_dim = interpolator.interpolate_bands(1)
        fs = FermiSurface.from_band_structure(new_bs, kpoint_dim)
        plotter = FermiSurfacePlotter(fs)
        plot = plotter.get_plot(plot_type="matplotlib")
        save_plot(plot, self.output_file)
        # plotter.plot(plot_type='plotly', interactive=True)
        # plotter.plot(plot_type='mayavi', interactive=False, filename=self.output_file)

    def test_integration_reciprocal(self):
        interpolator = Interpolator(self.band_structure)
        new_bs, kpoint_dim = interpolator.interpolate_bands(1)
        fs = FermiSurface.from_band_structure(new_bs, kpoint_dim, wigner_seitz=False)
        plotter = FermiSurfacePlotter(fs)
        plot = plotter.get_plot(plot_type="matplotlib")
        save_plot(plot, self.output_file)
        # plotter.plot(plot_type='plotly', interactive=True)
        # plotter.plot(plot_type='mayavi', interactive=False, filename=self.output_file)

    def test_integration_spin(self):
        interpolator = Interpolator(self.band_structure)
        new_bs, kpoint_dim = interpolator.interpolate_bands(1)
        fs = FermiSurface.from_band_structure(new_bs, kpoint_dim, wigner_seitz=False)
        plotter = FermiSurfacePlotter(fs)
        plot = plotter.get_plot(plot_type="matplotlib", spin=Spin.up)
        save_plot(plot, self.output_file)
        # plotter.plot(plot_type='plotly', interactive=True)
        # plotter.plot(plot_type='mayavi', interactive=False, filename=self.output_file)

    def tearDown(self):
        if self.output_file.exists():
            self.output_file.unlink()

        if (test_dir / "temp-plot.html").exists():
            (test_dir / "temp-plot.html").unlink()

        if (root_dir / "temp-plot.html").exists():
            (root_dir / "temp-plot.html").unlink()
