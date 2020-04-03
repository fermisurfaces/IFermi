from pathlib import Path
from ifermi.interpolator import Interpolater
from ifermi.plotter import FSPlotter
from ifermi.fermi_surface import FermiSurface
from pymatgen import Spin
from pymatgen.io.vasp.outputs import Vasprun
import warnings
warnings.simplefilter("ignore")

example_dir = Path("../../examples")
vr = Vasprun(example_dir / "MgB2/vasprun.xml")
bs = vr.get_band_structure()

interpolater = Interpolater(bs)
new_bs, kpoint_dim = interpolater.interpolate_bands(5)

fs = FermiSurface.from_band_structure(new_bs, kpoint_dim, wigner_seitz=False, spin=Spin.up)
plotter = FSPlotter(fs)
plotter.plot(plot_type='plotly')

# Make a three dimensional plot of the Reciprocal Cell

# fs = FermiSurface(new_bs, hdims, rlattvec, mu = 0.0, plot_wigner_seitz = False)

# plotter = FSPlotter(fs, rc, bz = None)

# plotter.fs_plot_data(plot_type = 'mayavi')

# Same as above but using plotly package

# fs = FermiSurface(new_bs, hdims, rlattvec, mu = 0.0, plot_wigner_seitz = False)

# plotter = FSPlotter(fs, rc, bz = None)

# plotter.fs_plot_data(plot_type = 'plotly')

# Two dimesnional slice plot

# fs = FermiSurface(new_bs, hdims, rlattvec, mu = 0.0, plot_wigner_seitz = False)

# plotter = FSPlotter2D(fs, plane_orig = (0., 0., 0.0), plane_norm = (0, 0, 1))

# plotter.fs2d_plot_data(plot_type = 'mpl')