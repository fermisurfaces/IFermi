from pathlib import Path

from monty.serialization import dumpfn
from pymatgen.io.vasp.outputs import Vasprun

from ifermi.interpolate import FourierInterpolator
from ifermi.surface import FermiSurface

if __name__ == "__main__":
    example_dir = Path("../../examples")
    vr = Vasprun(example_dir / "MgB2/vasprun.xml")
    bs = vr.get_band_structure()

    dumpfn(bs.structure, "structure.json.gz")

    interpolater = FourierInterpolator(bs)
    new_bs = interpolater.interpolate_bands(1)

    bs_data = {"bs": new_bs, "structure": bs.structure}
    dumpfn(bs_data, "bs_BaFe2As2.json.gz")

    fs = FermiSurface.from_band_structure(new_bs, wigner_seitz=True)
    dumpfn(fs, "fs_BaFe2As2_wigner.json.gz")
    dumpfn(fs.reciprocal_space, "rs_wigner.json.gz")

    fs = FermiSurface.from_band_structure(new_bs, wigner_seitz=False)
    dumpfn(fs, "fs_BaFe2As2_reciprocal.json.gz")
    dumpfn(fs.reciprocal_space, "rs_reciprocal.json.gz")
