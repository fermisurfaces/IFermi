from interpolator import Interpolater
from bulk_objects import FermiSurface, RecipCell
from plotter import *

from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure

if __name__ == '__main__':
	vr = Vasprun("../data_MgB2/vasprun.xml")
	bs = vr.get_band_structure()

	# increase interpolation factor to increase density of interpolated bandstructure
	interpolater = Interpolater(bs) 

	new_bs, hdims = interpolater.interpolate_bands(10)

	fs = FermiSurface(new_bs, hdims, new_bs.lattice_rec._matrix, mu = 0.1)

	rc = RecipCell(new_bs.lattice_rec._matrix)

	plotter = FSPlotter(fs, rc)

	plotter.fs_plot_data(plot_type = 'plotly', title_str = r"Fermi surface of $MgB_2$")




