from monty.serialization import *
from ifermi.fermi_surface import FermiSurface
from ifermi.interpolator import Interpolater

from pymatgen.io.vasp.outputs import Vasprun
import numpy as np

def writeFile():
    vr = Vasprun("/Users/amyjade/Documents/3rdYear/URAP/for_Sinead/IFermi/data_BaFe2As2/vasprun.xml")
    # vr = Vasprun("/Users/amyjade/Documents/IFermiPlottingTool/IFermi/data_MgB2/vasprun.xml")
    bs = vr.get_band_structure()

    interpolater = Interpolater(bs)

    new_bs, hdims, rlattvec = interpolater.interpolate_bands(1)

    # Make a three dimensional plot of the Brillioun zone

    fs = FermiSurface(new_bs.efermi, new_bs.structure, new_bs.bands, new_bs.kpoints, hdims, rlattvec, mu=0.0, plot_wigner_seitz=True)

    dumpfn(fs, '/Users/amyjade/PycharmProjects/FermiSurfaceTool/ifermi/data/BaFe2As2/fs_BaFe2As2.json')



if __name__ == '__main__':
    writeFile()