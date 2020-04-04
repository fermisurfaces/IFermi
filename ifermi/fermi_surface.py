"""
This module contains the classes and methods for creating iso-surface structures
from Pymatgen bandstrucutre objects. The iso-surfaces are found using the
Scikit-image package.
"""
import itertools
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
from monty.json import MSONable
from skimage import measure
from skimage.measure import marching_cubes_lewiner
from trimesh.intersections import slice_faces_plane

from ifermi.brillouin_zone import WignerSeitzCell, ReciprocalCell, ReciprocalSpace
from pymatgen import Spin, Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class FermiSurface(MSONable):
    """An object containing Fermi Surface data.

    Only stores information at k-points where energy(k) == Fermi energy.
    """

    def __init__(
        self,
        isosurfaces: Dict[Spin, List[Tuple[np.ndarray, np.ndarray]]],
        reciprocal_space: ReciprocalSpace,
        structure: Structure,
    ):
        """
        Get a Fermi Surface object.

        Args:
            isosurfaces: A dictionary containing a list of isosurfaces as
                ``(vertices, faces)`` for each spin channel.
            reciprocal_space: The reciprocal space associated with the Fermi surface.
            structure: The structure.
        """
        self.isosurfaces = isosurfaces
        self.reciprocal_space = reciprocal_space
        self.structure = structure
        self.n_surfaces = len(self.isosurfaces)

    @classmethod
    def from_band_structure(
        cls,
        band_structure: BandStructure,
        kpoint_dim: np.ndarray,
        mu: float = 0.0,
        wigner_seitz: bool = False,
        symprec: float = 0.001,
    ) -> "FermiSurface":
        """
        Args:
            band_structure: A band structure. The k-points must cover the full
                Brillouin zone (i.e., not just be the irreducible mesh). Use
                the ``ifermi.interpolator.Interpolator`` class to expand the k-points to
                the full Brillouin zone if required.
            kpoint_dim: The dimension of the grid in reciprocal space on which the
                energy eigenvalues are defined.
            mu: Energy offset from the Fermi energy at which the iso-surface is
               shape of the resulting iso-surface.
            wigner_seitz: Controls whether the cell is the Wigner-Seitz cell
                or the reciprocal unit cell parallelepiped.
            symprec: Symmetry precision for determining whether the structure is the
                standard primitive unit cell.
        """
        if np.product(kpoint_dim) != len(band_structure.kpoints):
            raise ValueError(
                "Number of k-points ({}) in band structure does not match number of "
                "k-points expected from mesh dimensions ({})".format(
                    len(band_structure.kpoints), np.product(kpoint_dim)
                )
            )

        band_structure = deepcopy(band_structure)  # prevent data getting overwritten

        structure = band_structure.structure
        fermi_level = band_structure.efermi + mu
        bands = band_structure.bands
        frac_kpoints = [k.frac_coords for k in band_structure.kpoints]
        frac_kpoints = np.array(frac_kpoints)

        if wigner_seitz:
            prim = get_prim_structure(structure, symprec=symprec)
            if not np.allclose(prim.lattice.matrix, structure.lattice.matrix, 1e-5):
                warnings.warn("Structure does not match expected primitive cell")

            reciprocal_space = WignerSeitzCell.from_structure(structure)
            bands, frac_kpoints, kpoint_dim = _expand_bands(
                bands, frac_kpoints, kpoint_dim
            )

        else:
            reciprocal_space = ReciprocalCell.from_structure(structure)

        kpoint_dim = tuple(kpoint_dim.astype(int))
        isosurfaces = compute_isosurfaces(
            bands,
            kpoint_dim,
            fermi_level,
            reciprocal_space,
        )

        return cls(isosurfaces, reciprocal_space, structure)

    def project_data(self, proj_plane: tuple):
        projected_band = []

        for i, band in enumerate(self.isosurfaces):
            verts = band[0]
            faces = band[1]
            projected_verts = []

            for vertex in verts:
                projected_verts.append(project(vertex, proj_plane))

            projected_band.append([projected_verts, faces])

        return projected_band

    @classmethod
    def from_dict(cls, d):
        """Returns FermiSurface object from dict."""
        fs = super().from_dict(d)
        isosurfaces = {Spin(int(k)): v for k, v in fs.isosurfaces.items()}
        return cls(isosurfaces, fs.reciprocal_space, fs.structure)

    def as_dict(self):
        """
        Get a json-serializable dict representation of FermiSurface.
        """
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "isosurfaces": {str(spin): iso for spin, iso in self.isosurfaces.items()},
            "structure": self.structure.as_dict(),
            "reciprocal_space": self.reciprocal_space.as_dict()
        }
        return d


def compute_isosurfaces(
    bands: Dict[Spin, np.ndarray],
    kpoint_dim: Tuple[int, int, int],
    fermi_level: float,
    reciprocal_space: ReciprocalSpace,
) -> Dict[Spin, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Compute the isosurfaces at a particular energy level.

    Args:
        bands: The band energies, given as a dictionary of ``{spin: energies}``, where
            energies has the shape (nbands, nkpoints).
        kpoint_dim: The k-point mesh dimensions.
        fermi_level: The energy at which to calculate the Fermi surface.
        reciprocal_space: The reciprocal space representation.

    Returns:
        A dictionary containing a list of isosurfaces as ``(vertices, faces)`` for
        each spin channel.
    """
    rlat = reciprocal_space.reciprocal_lattice

    spacing = 1 / (np.array(kpoint_dim) - 1)

    isosurfaces = {}
    for spin, ebands in bands.items():
        ebands -= fermi_level
        spin_isosurface = []

        for band in ebands:
            # check if band crosses fermi level
            if np.nanmax(band) > 0 > np.nanmin(band):
                band_data = band.reshape(kpoint_dim)
                verts, faces, _, _ = marching_cubes_lewiner(band_data, 0, spacing)

                if isinstance(reciprocal_space, WignerSeitzCell):
                    verts = np.dot(verts - 0.5, rlat) * 3
                    verts, faces = _trim_surface(reciprocal_space, verts, faces)
                else:
                    # convert coords to cartesian
                    verts = np.dot(verts - 0.5, rlat)

                spin_isosurface.append((verts, faces))

        isosurfaces[spin] = spin_isosurface

    return isosurfaces


def _trim_surface(
    wigner_seitz_cell: WignerSeitzCell,
    vertices: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trim the surface to remove parts outside the cell boundaries.

    Will add new triangles at the boundary edges as necessary to produce a smooth
    surface.

    Args:
        wigner_seitz_cell: The reciprocal space object.
        vertices: The surface vertices.
        faces: The surface faces.

    Returns:
        The trimmed surface as a tuple of ``(vertices, faces)``.
    """
    for center, normal in zip(wigner_seitz_cell.centers, wigner_seitz_cell.normals):
        vertices, faces = slice_faces_plane(vertices, faces, -normal, center)
    return vertices, faces


def _expand_bands(
    bands: Dict[Spin, np.ndarray], frac_kpoints: np.ndarray, kpoint_dim: np.ndarray
) -> Tuple[Dict[Spin, np.ndarray], np.ndarray, np.ndarray]:
    """
    Expand the band energies and k-points with periodic boundary conditions to form a
    3x3x3 supercell.

    Args:
        bands: The band energies, given as a dictionary of ``{spin: energies}``, where
            energies has the shape (nbands, nkpoints).
        frac_kpoints: The fractional k-point coordinates.
        kpoint_dim: The k-point mesh dimensions.

    Returns:
        The expanded band energies, k-points, and k-point mesh dimensions.
    """
    final_ebands = {}
    for spin, ebands in bands.items():
        super_ebands = []
        images = (-1, 0, 1)

        super_kpoints = np.array([], dtype=np.int64).reshape(0, 3)
        for i, j, k in itertools.product(images, images, images):
            k_image = frac_kpoints + [i, j, k]
            super_kpoints = np.concatenate((super_kpoints, k_image), axis=0)

        sort_idx = np.lexsort(
            (super_kpoints[:, 2], super_kpoints[:, 1], super_kpoints[:, 0])
        )
        final_kpoints = super_kpoints[sort_idx]

        for band in ebands:
            super_band = np.array([], dtype=np.int64)
            for _ in range(27):
                super_band = np.concatenate((super_band, band), axis=0)
            super_ebands.append(super_band[sort_idx])

        final_ebands[spin] = np.array(super_ebands)

    return final_ebands, final_kpoints, kpoint_dim * 3


def get_prim_structure(structure, symprec=0.01) -> Structure:
    """
    Get the primitive structure.

    Args:
        structure: The structure.
        symprec: The symmetry precision in Angstrom.

    Returns:
       The primitive cell as a pymatgen Structure object.
    """
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    return analyzer.get_primitive_standard_structure()


class FermiSurface2D(FermiSurface):
    def __init__(
        self,
        bs: BandStructure,
        hdims: list,
        rlattvec,
        slice_plane: tuple,
        contour,
        mu: float = 0.0,
        soc: bool = False,
    ) -> None:
        """
        Args:
            bs (BandStructure): A Pymatgen bandstructure object
            hdims (list): The dimension of the grid in reciprocal space on which the energy eigenvalues
                are defined.
            rlattvec (np.array): The reciprocal space lattice vectors. See
                pymatgen.electronic_structure.bandstructure.lattice_rec._matrix for format.
            slice_plane (tuple): The plane along which the surface is to be sliced. Only (0,0,1), (0,1,0)
                or (1,0,0) are currently supported.
            mu (float, optional): Enegy offset from the Fermi energy at which
                the iso-surface is defined. Useful for visualising the effect of
                dopants on the shape of the resulting iso-surface.
            kpoints (np.array): A numpy list of the kpoints in fractional coordinates
            soc (bool, optional): Set to True if the up and down spins are both to be plotted.
                Otherwsie, spins will be treated as degenerate and only one componenet will be
                plotted.
            is_spin_polarised (bool, optional): set to True if spin polarised.
            n_surfaces (int): Number of bands which cross the Fermi-Surface
        """

        self._mu = mu

        self._fermi_level = bs.efermi + mu

        self._kpoints = np.array([k.frac_coords for k in bs.kpoints])

        self._hdims = hdims

        dims = 2 * hdims + 1

        self._dims = dims

        self._k_dim = (dims[0], dims[1], dims[2])

        self._rlattvec = rlattvec

        self._slice_plane = slice_plane

        self.slice_data(bs, self._slice_plane)

        self.compute_isosurfaces(self._energies, self._fermi_level)

        self._n_bands = len(self._iso_surface)

        self._soc = soc

        self._structure = bs.structure

    def slice_data(self, bs, slice_plane: tuple):

        for spin in self._bands.keys():

            ebands = self._bands[spin]

            plane_bands = []

            dis_array = [
                plane_dist(np.append(proj_plane, -contour), i) for i in self._kpoints
            ]

            for i, j in enumerate(slice_array):
                if not j == 0:
                    if i == 0:
                        sort_indx = np.lexsort(dis_array[dis_array[:, 0].argsort()])
                        plane_mesh = [1, mesh[1], mesh[2]]
                    if i == 1:
                        sort_indx = np.lexsort(dis_array[dis_array[:, 1].argsort()])
                        plane_mesh = [mesh[0], 1, mesh[2]]

                    if i == 2:
                        sort_indx = np.lexsort(dis_array[dis_array[:, 2].argsort()])
                        plane_mesh = [mesh[0], mesh[1], 1]

            sorted_dist = dis_array(sort_index)
            sorted_energies = ebands(sort_indx)
            sorted_kpoints = self._kpoints(sort_index)

            for dist, index in enumerate(sorted_dist):
                while np.abs(dist - np.min(sorted_dist)) < 0.001:
                    plane_bands.append(sorted_energies[index])

            sort_idx = np.lexsort(
                (sorted_kpoints[:, 2], sorted_kpoints[:, 1], sorted_kpoints[:, 0],)
            )
            energies_sorted = sorted_energies[sort_idx]

            self._energies = energies.reshape(plane_mesh)

    def compute_isosurfaces(self, energies, contour: float):

        contours = measure.find_contours(energies, contour)

        self._contours = contour


def project(vector, plane):
    theta = np.array([0, 0, np.pi])
    a = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), np.sin(theta[0])],
            [0, -np.sin(theta[0]), np.cos(theta[0])],
        ]
    )

    b = np.array(
        [
            [np.cos(theta[1]), 0, -np.sin(theta[1])],
            [0, 1, 0],
            [np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )

    c = np.array(
        [
            [np.cos(theta[2]), np.sin(theta[2]), 0],
            [0, 1, 0],
            [np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )

    d = np.matmul(np.matmul(a, np.matmul(b, c)), vector)

    e = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    f = np.matmul(e, np.append(d, 1))

    b_x = f[0] / f[3]
    b_y = f[1] / f[3]

    return [b_x, b_y, 0]


def plane_dist(slice_plane, vertex):
    return (
        np.linalg.norm(
            slice_plane[0] * vertex[0]
            + slice_plane[1] * vertex[1]
            + slice_plane[2] * vertex[2]
            + slice_plane[3]
        )
    ) / (np.sqrt(slice_plane[0] ** 2 + slice_plane[1] ** 2 + slice_plane[2] ** 2))
