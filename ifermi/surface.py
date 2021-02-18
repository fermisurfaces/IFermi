"""Tools for creating iso-surface from BandStructure objects."""

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from monty.dev import requires
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.core import Spin

from ifermi.brillouin_zone import ReciprocalCell, WignerSeitzCell
from ifermi.interpolator import PeriodicLinearInterpolator

try:
    import mcubes
except ImportError:
    mcubes = None

try:
    import open3d
except ImportError:
    open3d = None

__all__ = [
    "Isosurface",
    "FermiSurface",
    "compute_isosurfaces",
    "trim_surface",
    "expand_bands",
    "decimate_mesh",
    "face_projections",
]


@dataclass
class Isosurface(MSONable):
    """
    An isosurface object contains a triangular mesh and surface properties.

    Attributes:
        vertices: A (n, 3) float array of the vertices in the isosurface.
        faces: A (m, 3) int array of the faces of the isosurface.
        band_idx: The band index to which the surface belongs.
        properties: An optional (m, 3, ...) float array containing face properties as
            scalars or vectors.
        dimensionality: The dimensionality of the surface.
        orientation: The orientation of the surface (for 1D and 2D surfaces only).
    """

    vertices: np.ndarray
    faces: np.ndarray
    band_idx: int
    properties: Optional[np.ndarray] = None
    dimensionality: Optional[str] = None
    orientation: Optional[Tuple[int, int, int]] = None

    @property
    def area(self) -> float:
        """Area of the isosurface."""
        from ifermi.analysis import isosurface_area

        return isosurface_area(self.vertices, self.faces)

    @property
    def has_properties(self) -> float:
        """Whether the surface has properties."""
        return self.properties is None

    def average_properties(
        self, norm: bool = False, axis: Optional[Tuple[int, int, int]] = None
    ) -> Union[float, np.ndarray]:
        """
        Average property across isosurface.

        Args:
            norm: Average the norm of the properties (vector properties only).
            axis: An axis to project the properties onto (vector properties only).

        Returns:
            The averaged property.
        """
        from ifermi.analysis import average_properties

        if not self.has_properties:
            raise ValueError("Isosurface does not have face properties.")

        properties = self.properties
        if axis is not None:
            properties = self.scalar_projection(axis)

        return average_properties(self.vertices, self.faces, properties, norm=norm)

    def scalar_projection(self, axis: Tuple[int, int, int]) -> np.ndarray:
        """
        Get scalar projection of properties onto axis.

        Args:
            axis: The axis to project onto.
        """
        if not self.has_properties:
            raise ValueError("Isosurface does not have face properties.")

        if self.properties.ndim != 2:
            raise ValueError("Isosurface does not have vector properties.")

        return np.dot(self.properties, axis)

    def sample_uniform(self, grid_size: float) -> np.ndarray:
        """
        Sample mesh faces uniformly.

        See the docstring for ``ifermi.analysis.sample_surface_uniform`` for more
        details.

        Args:
            grid_size: The grid size in Å^-1.

        Returns:
            A (n, ) int array containing the indices of uniformly spaced faces.
        """
        from ifermi.analysis import sample_surface_uniform

        return sample_surface_uniform(self.vertices, self.faces, grid_size)


@dataclass
class FermiSurface(MSONable):
    """An object containing Fermi Surface data.

    Stores information at k-points where energy(k) == Fermi energy.

    Args:
        isosurfaces: A dict containing a list of isosurfaces for each spin channel.
        reciprocal_space: A reciprocal space defining periodic boundary conditions.
        structure: The structure.
    """

    isosurfaces: Dict[Spin, List[Isosurface]]
    reciprocal_space: ReciprocalCell
    structure: Structure

    @property
    def n_surfaces(self) -> int:
        """Number of iso-surfaces in the Fermi surface."""
        return sum(map(len, self.isosurfaces.values()))

    @property
    def area(self) -> float:
        """Total area of all iso-surfaces in the Fermi surface."""
        return sum(map(sum, self.area_surfaces.values()))

    @property
    def area_surfaces(self) -> Dict[Spin, np.ndarray]:
        """Area of each iso-surface in the Fermi surface."""
        return {k: np.array([i.area for i in v]) for k, v in self.isosurfaces.items()}

    @property
    def has_properties(self) -> bool:
        return all(
            [all([i.has_properties for i in s]) for s in self.isosurfaces.values()]
        )

    @classmethod
    def from_band_structure(
        cls,
        band_structure: BandStructure,
        mu: float = 0.0,
        wigner_seitz: bool = False,
        decimate_factor: Optional[float] = None,
        decimate_method: str = "quadric",
        smooth: bool = False,
        projection_data: Optional[Dict[Spin, np.ndarray]] = None,
        projection_kpoints: Optional[np.ndarray] = None,
        calculate_dimensionality: bool = False,
    ) -> "FermiSurface":
        """
        Create a FermiSurface from a pymatgen band structure object.

        Args:
            band_structure: A band structure. The k-points must cover the full
                Brillouin zone (i.e., not just be the irreducible mesh). Use
                the ``ifermi.interpolator.Interpolator`` class to expand the k-points to
                the full Brillouin zone if required.
            mu: Energy offset from the Fermi energy at which the iso-surface is
                calculated.
            wigner_seitz: Controls whether the cell is the Wigner-Seitz cell
                or the reciprocal unit cell parallelepiped.
            decimate_factor: If method is "quadric", factor is the scaling factor by
                which to reduce the number of faces. I.e., final # faces = initial
                # faces * factor. If method is "cluster", factor is the voxel size in
                which to cluster points. Default is None (no decimation).
            decimate_method: Algorithm to use for decimation. Options are "quadric"
                or "cluster".
            smooth: If True, will smooth resulting isosurface. Requires PyMCubes. See
                compute_isosurfaces for more information.
            projection_data: A property to project onto the Fermi surface. It should be
                given as a dict of ``{spin: properties}``, where properties is numpy
                array with shape (nbands, nkpoints, ...). The number of bands should
                equal the number of bands in the band structure but the k-point mesh
                can be different. Must be used in combination with
                ``kpoints``.
            projection_kpoints: The k-points on which the data is generated.
                Must be used in combination with ``data``.
            calculate_dimensionality: Whether to calculate isosurface dimensionalities.

        Returns:
            A Fermi surface.
        """
        from ifermi.kpoints import get_kpoint_mesh_dim, kpoints_from_bandstructure

        band_structure = deepcopy(band_structure)  # prevent data getting overwritten

        structure = band_structure.structure
        fermi_level = band_structure.efermi + mu
        bands = band_structure.bands
        kpoints = kpoints_from_bandstructure(band_structure)

        kpoint_dim = get_kpoint_mesh_dim(kpoints)
        if np.product(kpoint_dim) != len(kpoints):
            raise ValueError(
                "Number of k-points ({}) in band structure does not match number of "
                "k-points expected from mesh dimensions ({})".format(
                    len(band_structure.kpoints), np.product(kpoint_dim)
                )
            )

        if wigner_seitz:
            reciprocal_space = WignerSeitzCell.from_structure(structure)
        else:
            reciprocal_space = ReciprocalCell.from_structure(structure)

        interpolator = None
        if projection_data is not None and projection_kpoints is not None:
            interpolator = PeriodicLinearInterpolator(
                projection_kpoints, projection_data
            )
        elif projection_data is not None or projection_kpoints is not None:
            raise ValueError("Both data and kpoints must be specified.")

        bands, kpoints = expand_bands(
            bands, kpoints, supercell_dim=(3, 3, 3), center=(1, 1, 1)
        )
        isosurfaces = compute_isosurfaces(
            bands,
            kpoints,
            fermi_level,
            reciprocal_space,
            decimate_factor=decimate_factor,
            decimate_method=decimate_method,
            smooth=smooth,
            calculate_dimensionality=calculate_dimensionality,
            projection_interpolator=interpolator,
        )

        return cls(isosurfaces, reciprocal_space, structure)

    def get_fermi_slice(
        self, plane_normal: Tuple[int, int, int], distance: float = 0
    ) -> "FermiSlice":
        """
        Get a slice through the Fermi surface.

        Slice defined by the intersection of a plane with the Fermi surface.

        Args:
            plane_normal: The plane normal in fractional indices. E.g., ``(1, 0, 0)``.
            distance: The distance from the center of the Brillouin zone (the Gamma
                point).

        Returns:
            The Fermi slice.
        """
        from ifermi.slice import FermiSlice

        return FermiSlice.from_fermi_surface(self, plane_normal, distance=distance)

    @classmethod
    def from_dict(cls, d) -> "FermiSurface":
        """Return FermiSurface object from dict."""
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
    kpoints: np.ndarray,
    fermi_level: float,
    reciprocal_space: ReciprocalCell,
    decimate_factor: Optional[float] = None,
    decimate_method: str = "quadric",
    smooth: bool = False,
    calculate_dimensionality: bool = False,
    projection_interpolator: Optional[PeriodicLinearInterpolator] = None,
) -> Dict[Spin, List[Isosurface]]:
    """
    Compute the isosurfaces at a particular energy level.

    Args:
        bands: The band energies, given as a dictionary of ``{spin: energies}``, where
            energies has the shape (nbands, nkpoints).
        kpoints: The k-points in fractional coordinates.
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
        calculate_dimensionality: Whether to calculate isosurface dimensionality.
        projection_interpolator: An interpolator class for interpolating properties
            onto the surface. If ``None``, no properties will be calculated.

    Returns:
        A dictionary containing a list of isosurfaces for each spin channel.
    """
    from ifermi.kpoints import get_kpoint_mesh_dim, get_kpoint_spacing

    # sort k-points to be in the correct order
    order = np.lexsort((kpoints[:, 2], kpoints[:, 1], kpoints[:, 0]))
    kpoints = kpoints[order]
    bands = {s: b[:, order] for s, b in bands.items()}

    kpoint_dim = get_kpoint_mesh_dim(kpoints)
    spacing = get_kpoint_spacing(kpoints)
    reference = np.min(kpoints, axis=0)

    isosurfaces = {}
    for spin, ebands in bands.items():
        ebands -= fermi_level
        spin_isosurface = []

        for band_idx, energies in enumerate(ebands):
            if not np.nanmax(energies) > 0 > np.nanmin(energies):
                # if band doesn't cross the Fermi level then skip it
                continue

            band_isosurfaces = _calculate_band_isosurfaces(
                spin,
                band_idx,
                energies,
                kpoint_dim,
                spacing,
                reference,
                reciprocal_space,
                decimate_factor,
                decimate_method,
                smooth,
                calculate_dimensionality,
                projection_interpolator,
            )
            spin_isosurface.extend(band_isosurfaces)

        isosurfaces[spin] = spin_isosurface

    return isosurfaces


def _calculate_band_isosurfaces(
    spin: Spin,
    band_idx: int,
    energies: np.ndarray,
    kpoint_dim: Tuple[int, int, int],
    spacing: np.ndarray,
    reference: np.ndarray,
    reciprocal_space: ReciprocalCell,
    decimate_factor: Optional[float],
    decimate_method: str,
    smooth: bool,
    calculate_dimensionality: bool,
    projection_interpolator: Optional[PeriodicLinearInterpolator],
):
    """Helper function to calculate the connected isosurfaces for a band."""
    from skimage.measure import marching_cubes

    from ifermi.analysis import (
        connected_subsurfaces,
        equivalent_surfaces,
        isosurface_dimensionality,
    )

    rlat = reciprocal_space.reciprocal_lattice

    if smooth and mcubes is None:
        smooth = False
        warnings.warn("Smoothing disabled, install PyMCubes to enable smoothing.")

    if decimate_factor is not None and open3d is None:
        decimate_factor = None
        warnings.warn("Decimation disabled, install open3d to enable decimation.")

    band_data = energies.reshape(kpoint_dim)

    if smooth:
        smoothed_band_data = mcubes.smooth(band_data)
        verts, faces = mcubes.marching_cubes(smoothed_band_data, 0)
        # have to manually set spacing with PyMCubes
        verts *= spacing
        # comes out as np.uint64, but trimesh doesn't like this
        faces = faces.astype(np.int32)
    else:
        verts, faces, _, _ = marching_cubes(band_data, 0, spacing=spacing)

    if decimate_factor:
        verts, faces = decimate_mesh(
            verts, faces, decimate_factor, method=decimate_method
        )

    verts += reference

    # break the isosurface into connected subsurfaces
    subsurfaces = connected_subsurfaces(verts, faces)

    if calculate_dimensionality:
        # calculate dimensionality of periodically equivalent surfaces.
        dimensionalities = {}
        mapping = equivalent_surfaces([s[0] for s in subsurfaces])
        for idx in mapping:
            dimensionalities[idx] = isosurface_dimensionality(*subsurfaces[idx])
    else:
        dimensionalities = None
        mapping = np.zeros(len(subsurfaces))

    isosurfaces = []
    dimensionality = None
    orientation = None
    projections = None
    for (subverts, subfaces), idx in zip(subsurfaces, mapping):
        # convert vertices to cartesian coordinates
        subverts = np.dot(subverts, rlat)
        subverts, subfaces = trim_surface(reciprocal_space, subverts, subfaces)

        if len(subverts) == 0:
            # skip surfaces that do not enter the reciprocal space boundaries
            continue

        if calculate_dimensionality:
            dimensionality, orientation = dimensionalities[idx]

        if projection_interpolator is not None:
            projections = face_projections(
                projection_interpolator, subverts, subfaces, band_idx, spin, rlat
            )

        isosurface = Isosurface(
            vertices=subverts,
            faces=subfaces,
            band_idx=band_idx,
            projections=projections,
            dimensionality=dimensionality,
            orientation=orientation,
        )
        isosurfaces.append(isosurface)
    return isosurfaces


def trim_surface(
    reciprocal_cell: ReciprocalCell, vertices: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trim the surface to remove parts outside the cell boundaries.

    Will add new triangles at the boundary edges as necessary to produce a smooth
    surface.

    Args:
        reciprocal_cell: The reciprocal space object.
        vertices: The surface vertices.
        faces: The surface faces.

    Returns:
        The trimmed surface as a tuple of ``(vertices, faces)``.
    """
    from trimesh.intersections import slice_faces_plane

    for center, normal in zip(reciprocal_cell.centers, reciprocal_cell.normals):
        vertices, faces = slice_faces_plane(vertices, faces, -normal, center)
    return vertices, faces


def expand_bands(
    bands: Dict[Spin, np.ndarray],
    frac_kpoints: np.ndarray,
    supercell_dim: Tuple[int, int, int] = (3, 3, 3),
    center: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Dict[Spin, np.ndarray], np.ndarray]:
    """
    Expand the band energies and k-points with periodic boundary conditions.

    Args:
        bands: The band energies, given as a dictionary of ``{spin: energies}``, where
            energies has the shape (nbands, nkpoints).
        frac_kpoints: The fractional k-point coordinates.
        supercell_dim: The supercell mesh dimensions.
        center: The cell on which the supercell is centered.

    Returns:
        The expanded band energies and k-points.
    """
    final_ebands = {}
    nk = len(frac_kpoints)
    ncells = np.product(supercell_dim)

    final_kpoints = np.tile(frac_kpoints, (ncells, 1))
    for n, (i, j, k) in enumerate(np.ndindex(supercell_dim)):
        final_kpoints[n * nk : (n + 1) * nk] += [i, j, k]
    final_kpoints -= center

    for spin, ebands in bands.items():
        final_ebands[spin] = np.tile(ebands, (1, ncells))

    return final_ebands, final_kpoints


def face_projections(
    interpolator: PeriodicLinearInterpolator,
    vertices: np.ndarray,
    faces: np.ndarray,
    band_idx: int,
    spin: Spin,
    reciprocal_lattice: np.ndarray,
) -> np.ndarray:
    """
    Interpolate properties data onto the Fermi surfaces.

    Args:
        interpolator: Periodic interpolator for projection data.
        vertices: The vertices of the surface.
        faces: The faces of the surface.
        band_idx: The band that the isosurface belongs to.
        spin: The spin channel the isosurface belongs to.
        reciprocal_lattice: Reciprocal lattice matrix

    Returns:
        The interpolated properties at the center of each face in the isosurface.
    """
    from ifermi.kpoints import kpoints_to_first_bz

    # get the center of each of face in fractional coords
    inv_lattice = np.linalg.inv(reciprocal_lattice)
    face_verts = np.dot(vertices, inv_lattice)[faces]
    centers = face_verts.mean(axis=1)

    # convert to 1st BZ
    centers = kpoints_to_first_bz(centers)

    # get interpolated properties at center of faces
    band_idxs = np.full(len(centers), band_idx)
    return interpolator.interpolate(spin, band_idxs, centers)


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
