from interpolator import Interpolater
from fermi_surface import FermiSurface
from brillouin_zone import BrillouinZone, RecipCell
from plotter import *

from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure

if __name__ == '__main__':
	vr = Vasprun("../data_MgB2/vasprun.xml")
	bs = vr.get_band_structure()

	# increase interpolation factor to increase density of interpolated bandstructure
	interpolater = Interpolater(bs) 

	new_bs, hdims, rlattvec = interpolater.interpolate_bands(10)

	rc = RecipCell(rlattvec)

	bz = BrillouinZone(rlattvec)

	# Make a three dimensional plot of the Brillioun zone

	# fs = FermiSurface(new_bs, hdims, rlattvec, mu = 0.0, plot_wigner_seitz = True)

	# plotter = FSPlotter(fs, rc = None, bz = bz)

	# plotter.fs_plot_data(plot_type = 'mayavi')

	# Make a three dimensional plot of the Reciprocal Cell 

	# fs = FermiSurface(new_bs, hdims, rlattvec, mu = 0.0, plot_wigner_seitz = False)

	# plotter = FSPlotter(fs, rc, bz = None)

	# plotter.fs_plot_data(plot_type = 'mayavi')

	# Same as above but using plotly package

	# fs = FermiSurface(new_bs, hdims, rlattvec, mu = 0.0, plot_wigner_seitz = False)

	# plotter = FSPlotter(fs, rc, bz = None)

	# plotter.fs_plot_data(plot_type = 'plotly')

	# Two dimesnional slice plot

	fs = FermiSurface(new_bs, hdims, rlattvec, mu = 0.0, plot_wigner_seitz = False)

	plotter = FSPlotter2D(fs, plane_orig = (0., 0., 0.0), plane_norm = (0, 0, 1))

	plotter.fs2d_plot_data(plot_type = 'mpl')

