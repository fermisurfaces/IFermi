"""
This module implements a class to perform band structure interpolation using
BolzTraP2. Developed by Alex Ganose
(https://gist.github.com/utf/160118f74d8a58fc9abf9c1c3f52a384).
"""

import multiprocessing
from collections import defaultdict
from typing import Optional

import numpy as np
from BoltzTraP2 import fite, sphere
from BoltzTraP2.units import Angstrom, eV
from monty.json import MSONable
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.io.ase import AseAtomsAdaptor
from spglib import spglib


class Interpolator(MSONable):
    """Takes a pymatgen BandStructure object and interpolates the bands to
    create a denser mesh. This is done using Boltzstrap2, a module which is able
    to interpolate bands using Fourier coefficients. Implementation taken from
    Alex Ganose's AMSET Interpolator class.

    Args:
        band_structure (BandStructure): The Bandstructure object to be
            interpolated
        soc (bool): Whether the band structure is calculated using spin-orbit
            coupling.
        magmom: Magnetic moments of the atoms.
        mommat: Momentum matrix, as supported by BoltzTraP2.
    """

    def __init__(
        self,
        band_structure: BandStructure,
        soc: bool = False,
        magmom: Optional[np.ndarray] = None,
        mommat: Optional[np.ndarray] = None,
    ):
        self._band_structure = band_structure
        self._soc = soc
        self._spins = self._band_structure.bands.keys()
        self._lattice_matrix = band_structure.structure.lattice.matrix.T * Angstrom
        self._projection_coefficients = defaultdict(dict)

        self._kpoints = np.array([k.frac_coords for k in band_structure.kpoints])
        self._atoms = AseAtomsAdaptor.get_atoms(band_structure.structure)

        self._magmom = magmom
        self._mommat = mommat
        self._structure = band_structure.structure

    def interpolate_bands(
        self,
        interpolation_factor: float = 5,
        energy_cutoff: Optional[float] = None,
        nworkers: int = -1,
    ):
        """Gets a pymatgen band structure.

        Note, the interpolation mesh is determined using by ``interpolate_factor``
        option in the ``Interpolator`` constructor.

        The degree of parallelization is controlled by the ``nworkers`` option.

        Args:
            interpolation_factor: The factor by which the band structure will
                be interpolated.
            energy_cutoff: The energy cut-off to determine which bands are
                included in the interpolation. If the energy of a band falls
                within the cut-off at any k-point it will be included. For
                metals the range is defined as the Fermi level ± energy_cutoff.
                For gapped materials, the energy range is from the VBM -
                energy_cutoff to the CBM + energy_cutoff.
            nworkers: The number of processors used to perform the
                interpolation. If set to ``-1``, the number of workers will
                be set to the number of CPU cores.

        Returns:
            The interpolated electronic structure.
        """

        coefficients = {}

        equivalences = sphere.get_equivalences(
            atoms=self._atoms,
            nkpt=self._kpoints.shape[0] * interpolation_factor,
            magmom=self._magmom,
        )

        # get the interpolation mesh used by BoltzTraP2
        interpolation_mesh = 2 * np.max(np.abs(np.vstack(equivalences)), axis=0) + 1

        for spin in self._spins:
            energies = self._band_structure.bands[spin] * eV
            data = DFTData(
                self._kpoints, energies, self._lattice_matrix, mommat=self._mommat
            )
            coefficients[spin] = fite.fitde3D(data, equivalences)
        is_metal = self._band_structure.is_metal()

        nworkers = multiprocessing.cpu_count() if nworkers == -1 else nworkers

        # determine energy cutoffs
        if energy_cutoff and is_metal:
            min_e = self._band_structure.efermi - energy_cutoff
            max_e = self._band_structure.efermi + energy_cutoff

        elif energy_cutoff:
            min_e = self._band_structure.get_vbm()["energy"] - energy_cutoff
            max_e = self._band_structure.get_cbm()["energy"] + energy_cutoff

        else:
            min_e = min(
                [self._band_structure.bands[spin].min() for spin in self._spins]
            )
            max_e = max(
                [self._band_structure.bands[spin].max() for spin in self._spins]
            )

        energies = {}
        new_vb_idx = {}
        for spin in self._spins:
            ibands = np.any(
                (self._band_structure.bands[spin] > min_e)
                & (self._band_structure.bands[spin] < max_e),
                axis=1,
            )

            energies[spin] = fite.getBTPbands(
                equivalences,
                coefficients[spin][ibands],
                self._lattice_matrix,
                nworkers=nworkers,
            )[0]

            # boltztrap2 gives energies in Rydberg, convert to eV
            energies[spin] /= eV

            if not is_metal:
                vb_energy = self._band_structure.get_vbm()["energy"]
                spin_bands = self._band_structure.bands[spin]
                below_vbm = np.any(spin_bands < vb_energy, axis=1)
                spin_vb_idx = np.max(np.where(below_vbm)[0])

                # need to know the index of the valence band after discounting
                # bands during the interpolation. As ibands is just a list of
                # True/False, we can count the number of Trues up to
                # and including the VBM to get the new number of valence bands
                new_vb_idx[spin] = sum(ibands[: spin_vb_idx + 1]) - 1

        efermi = self._band_structure.efermi

        atoms = AseAtomsAdaptor().get_atoms(self._band_structure.structure)
        mapping, grid = spglib.get_ir_reciprocal_mesh(
            interpolation_mesh, atoms, symprec=0.1
        )
        kpoints = grid / interpolation_mesh

        # sort energies so they have the same order as the k-points generated by spglib
        sort_idx = sort_boltztrap_to_spglib(kpoints)
        energies = {s: ener[:, sort_idx] for s, ener in energies.items()}

        rlat = self._band_structure.structure.lattice.reciprocal_lattice
        interp_band_structure = BandStructure(
            kpoints, energies, rlat, efermi, structure=self._structure
        )

        return interp_band_structure, interpolation_mesh


class DFTData(object):
    """DFTData object used for BoltzTraP2 interpolation.

    Note that the units used by BoltzTraP are different to those used by VASP.

    Args:
        kpoints: The k-points in fractional coordinates.
        energies: The band energies in Hartree, formatted as (nbands, nkpoints).
        lattice_matrix: The lattice matrix in Bohr^3.
        mommat: The band structure derivatives.
    """

    def __init__(
        self,
        kpoints: np.ndarray,
        energies: np.ndarray,
        lattice_matrix: np.ndarray,
        mommat: Optional[np.ndarray] = None,
    ):
        self.kpoints = kpoints
        self.ebands = energies
        self.lattice_matrix = lattice_matrix
        self.volume = np.abs(np.linalg.det(self.lattice_matrix))
        self.mommat = mommat

    def get_lattvec(self) -> np.ndarray:
        """Get the lattice matrix. This method is required by BoltzTraP2."""
        return self.lattice_matrix


def sort_boltztrap_to_spglib(kpoints):
    sort_idx = np.lexsort(
        (
            kpoints[:, 2],
            kpoints[:, 2] < 0,
            kpoints[:, 1],
            kpoints[:, 1] < 0,
            kpoints[:, 0],
            kpoints[:, 0] < 0,
        )
    )
    boltztrap_kpoints = kpoints[sort_idx]

    sort_idx = np.lexsort(
        (
            boltztrap_kpoints[:, 0],
            boltztrap_kpoints[:, 0] < 0,
            boltztrap_kpoints[:, 1],
            boltztrap_kpoints[:, 1] < 0,
            boltztrap_kpoints[:, 2],
            boltztrap_kpoints[:, 2] < 0,
        )
    )
    return sort_idx
