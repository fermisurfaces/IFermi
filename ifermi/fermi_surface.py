"""
This module contains the classes and methods for creating iso-surface structures
from Pymatgen bandstrucutre objects. The iso-surfaces are found using the
Scikit-image package.
"""

import itertools
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from monty.dev import requires
from monty.json import MSONable
from pymatgen import Spin, Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from skimage.measure import marching_cubes
from trimesh import Trimesh
from trimesh.intersections import mesh_multiplane, slice_faces_plane

from ifermi.brillouin_zone import ReciprocalCell, ReciprocalSlice, WignerSeitzCell

try:
    import mcubes
except ImportError:
    mcubes = None

try:
    import open3d
except ImportError:
    open3d = None


@dataclass
class FermiSlice(MSONable):
    """
    A 2D slice through a Fermi surface.

    Args:
        slices: The slices for each spin channel. Given as a dictionary of
            ``{spin: spin_slices}`` where spin_slices is a List of numpy arrays, each
            with the shape ``(n_lines, 2, 2)``.
        reciprocal_slice: The reciprocal slice defining the intersection of the
            plane with the Brillouin zone edges.
        structure: The structure.

    """

    slices: Dict[Spin, List[np.ndarray]]
    reciprocal_slice: ReciprocalSlice
    structure: Structure

    @classmethod
    def from_dict(cls, d) -> "FermiSlice":
        """Returns FermiSurface object from dict."""
        fs = super().from_dict(d)
        fs.slices = {Spin(int(k)): v for k, v in fs.slices.items()}
        return fs

    def as_dict(self) -> dict:
        """Get a json-serializable dict representation of FermiSurface."""
        d = super().as_dict()
        d["slices"] = {str(spin): iso for spin, iso in self.slices.items()}
        return d


@dataclass
class FermiSurface(MSONable):
    """An object containing Fermi Surface data.

    Only stores information at k-points where energy(k) == Fermi energy.

    Args:
        isosurfaces: A dictionary containing a list of isosurfaces as ``(vertices,
            faces)`` for each spin channel.
        reciprocal_space: The reciprocal space associated with the Fermi surface.
        structure: The structure.

    """

    isosurfaces: Dict[Spin, List[Tuple[np.ndarray, np.ndarray]]]
    reciprocal_space: ReciprocalCell
    structure: Structure

    @property
    def n_surfaces(self) -> int:
        return sum(map(len, self.isosurfaces.values()))

    @classmethod
    def from_band_structure(
        cls,
        band_structure: BandStructure,
        kpoint_dim: np.ndarray,
        mu: float = 0.0,
        wigner_seitz: bool = False,
        symprec: float = 0.001,
        decimate_factor: Optional[float] = None,
        decimate_method: str = "quadric",
        smooth: bool = False,
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
                calculated.
            wigner_seitz: Controls whether the cell is the Wigner-Seitz cell
                or the reciprocal unit cell parallelepiped.
            symprec: Symmetry precision for determining whether the structure is the
                standard primitive unit cell.
            decimate_factor: If method is "quadric", factor is the scaling factor by
                which to reduce the number of faces. I.e., final # faces = initial
                # faces * factor. If method is "cluster", factor is the voxel size in
                which to cluster points. Default is None (no decimation).
            decimate_method: Algorithm to use for decimation. Options are "quadric"
                or "cluster".
            smooth: If True, will smooth resulting isosurface. Requires PyMCubes. See
                compute_isosurfaces for more information.
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
        kpoints = np.array([k.frac_coords for k in band_structure.kpoints])

        # sort k-points to be in the correct order
        order = np.lexsort((kpoints[:, 2], kpoints[:, 1], kpoints[:, 0]))
        kpoints = kpoints[order]
        bands = {s: b[:, order] for s, b in bands.items()}

        if wigner_seitz:
            prim = get_prim_structure(structure, symprec=symprec)
            if not np.allclose(prim.lattice.matrix, structure.lattice.matrix, 1e-5):
                warnings.warn("Structure does not match expected primitive cell")

            reciprocal_space = WignerSeitzCell.from_structure(structure)
            bands, kpoints, kpoint_dim = _expand_bands(bands, kpoints, kpoint_dim)

        else:
            reciprocal_space = ReciprocalCell.from_structure(structure)

        kpoint_dim = tuple(kpoint_dim.astype(int))
        isosurfaces = compute_isosurfaces(
            bands,
            kpoint_dim,
            fermi_level,
            reciprocal_space,
            decimate_factor=decimate_factor,
            decimate_method=decimate_method,
            smooth=smooth,
        )

        return cls(isosurfaces, reciprocal_space, structure)

    def get_fermi_slice(
        self, plane_normal: Tuple[int, int, int], distance: float = 0
    ) -> FermiSlice:
        """
        Get a slice through the Fermi surface, defined by the intersection of a plane
        with the fermi surface.

        Args:
            plane_normal: The plane normal in fractional indices. E.g., ``(1, 0, 0)``.
            distance: The distance from the center of the Brillouin zone (the Gamma
                point).

        Returns:
            The Fermi slice.

        """
        cart_normal = np.dot(plane_normal, self.reciprocal_space.reciprocal_lattice)
        cart_origin = cart_normal * distance

        slices = {}
        for spin, spin_isosurfaces in self.isosurfaces.items():
            spin_slices = []

            for verts, faces in spin_isosurfaces:
                mesh = Trimesh(vertices=verts, faces=faces)
                lines = mesh_multiplane(mesh, cart_origin, cart_normal, [0])[0][0]
                spin_slices.append(lines)

            slices[spin] = spin_slices

        reciprocal_slice = self.reciprocal_space.get_reciprocal_slice(
            plane_normal, distance
        )

        return FermiSlice(slices, reciprocal_slice, self.structure)

    @classmethod
    def from_dict(cls, d) -> "FermiSurface":
        """Returns FermiSurface object from dict."""
        fs = super().from_dict(d)
        fs.isosurfaces = {Spin(int(k)): v for k, v in fs.isosurfaces.items()}
        return fs

    def as_dict(self) -> dict:
        """Get a json-serializable dict representation of FermiSurface."""
        d = super().as_dict()
        d["isosurfaces"] = {str(spin): iso for spin, iso in self.isosurfaces.items()}
        return d


def compute_isosurfaces(
    bands: Dict[Spin, np.ndarray],
    kpoint_dim: Tuple[int, int, int],
    fermi_level: float,
    reciprocal_space: ReciprocalCell,
    decimate_factor: Optional[float] = None,
    decimate_method: str = "quadric",
    smooth: bool = False,
) -> Dict[Spin, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Compute the isosurfaces at a particular energy level.

    Args:
        bands: The band energies, given as a dictionary of ``{spin: energies}``, where
            energies has the shape (nbands, nkpoints).
        kpoint_dim: The k-point mesh dimensions.
        fermi_level: The energy at which to calculate the Fermi surface.
        reciprocal_space: The reciprocal space representation.
        decimate_factor: If method is "quadric", factor is the scaling factor by which
            to reduce the number of faces. I.e., final # faces = initial # faces *
            factor. If method is "cluster", factor is the voxel size in which to
            cluster points. Default is None (no decimation).
        decimate_method: Algorithm to use for decimation. Options are "quadric" or
            "cluster".
        smooth: If True, will smooth resulting isosurface. Requires PyMCubes. Smoothing
            algorithm will use constrained smoothing algorithm to preserve fine details
            if input dimension is lower than (500, 500, 500), otherwise will apply a
            Gaussian filter.

    Returns:
        A dictionary containing a list of isosurfaces as ``(vertices, faces)`` for
        each spin channel.
    """
    rlat = reciprocal_space.reciprocal_lattice

    spacing = 1 / (np.array(kpoint_dim) - 1)

    if smooth and mcubes is None:
        smooth = False
        warnings.warn("Smoothing disabled, install PyMCubes to enable smoothing.")

    if decimate_factor is not None and open3d is None:
        decimate_factor = None
        warnings.warn("Decimation disabled, install open3d to enable decimation.")

    isosurfaces = {}
    for spin, ebands in bands.items():
        ebands -= fermi_level
        spin_isosurface = []

        for band in ebands:
            # check if band crosses fermi level
            if np.nanmax(band) > 0 > np.nanmin(band):
                band_data = band.reshape(kpoint_dim)

                if smooth:
                    smoothed_band_data = mcubes.smooth(band_data)
                    # and outputs embedding array with values 0 and 1
                    verts, faces = mcubes.marching_cubes(smoothed_band_data, 0)
                    # have to manually set spacing with PyMCubes
                    verts = verts * spacing
                    # comes out as np.uint64, but trimesh doesn't like this
                    faces = faces.astype(np.int32)
                else:
                    verts, faces, _, _ = marching_cubes(band_data, 0, spacing=spacing)

                if decimate_factor:
                    verts, faces = decimate_mesh(
                        verts, faces, decimate_factor, method=decimate_method
                    )

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
    wigner_seitz_cell: WignerSeitzCell, vertices: np.ndarray, faces: np.ndarray
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
    final_kpoints = None
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


@requires(open3d, "open3d package is required for mesh decimation")
def decimate_mesh(vertices: np.ndarray, faces: np.ndarray, factor, method="quadric"):
    """Decimate mesh to reduce the number of triangles and vertices.

    The open3d package is required for decimation.

    Args:
        vertices: The mesh vertices.
        faces: The mesh faces.
        factor: If method is "quadric", factor is the scaling factor by which to
            reduce the number of faces. I.e., final # faces = initial # faces * factor.
            If method is "cluster", factor is the voxel size in which to cluster points.
        method: Algorithm to use for decimation. Options are "quadric" or "cluster".
    """
    # convert mesh to open3d format
    o3d_verts = open3d.utility.Vector3dVector(vertices)
    o3d_faces = open3d.utility.Vector3iVector(faces)
    o3d_mesh = open3d.geometry.TriangleMesh(o3d_verts, o3d_faces)

    # decimate mesh
    if method == "quadric":
        n_target_triangles = int(len(faces) * factor)
        o3d_new_mesh = o3d_mesh.simplify_quadric_decimation(n_target_triangles)
    else:
        cluster_type = open3d.geometry.SimplificationContraction.Quadric
        o3d_new_mesh = o3d_mesh.simplify_vertex_clustering(factor, cluster_type)

    return np.array(o3d_new_mesh.vertices), np.array(o3d_new_mesh.triangles)
