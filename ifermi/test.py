from ifermi.interpolator import Interpolater
from ifermi.plotter import *

from pymatgen.io.vasp.outputs import Vasprun

if __name__ == '__main__':
	vr = Vasprun("/Users/amyjade/Documents/3rdYear/URAP/for_Sinead/IFermi/data_BaFe2As2/vasprun.xml")
	#vr = Vasprun("/Users/amyjade/Documents/IFermiPlottingTool/IFermi/data_MgB2/vasprun.xml")
	bs = vr.get_band_structure()

	# increase interpolation factor to increase density of interpolated bandstructure
	interpolater = Interpolater(bs)

	new_bs, hdims, rlattvec= interpolater.interpolate_bands(10)

	# Make a three dimensional plot of the Brillioun zone

	fs = FermiSurface(new_bs.efermi, new_bs.structure, new_bs.bands, hdims, rlattvec, mu = 0.0, plot_wigner_seitz = True)

	plotter = FSPlotter(fs)

	plotter.fs_plot_data(plot_type = 'mayavi')

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