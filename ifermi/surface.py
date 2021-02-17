"""Tools for creating iso-surface from BandStructure objects."""

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from monty.dev import requires
from monty.json import MSONable

from ifermi.analysis import SurfaceProperties
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.core import Spin

from ifermi.brillouin_zone import ReciprocalCell, WignerSeitzCell

try:
    import mcubes
except ImportError:
    mcubes = None

try:
    import open3d
except ImportError:
    open3d = None

__all__ = [
    "FermiSurface",
    "compute_isosurfaces",
    "trim_surface",
    "expand_bands",
    "decimate_mesh",
    "get_fermi_surface_projection",
]


# @dataclass
# class Isosurface(MSONable):
#     vertices: np.ndarray
#
#


@dataclass
class FermiSurface(MSONable):
    """An object containing Fermi Surface data.

    Only stores information at k-points where energy(k) == Fermi energy.

    Args:
        isosurfaces: A dictionary containing a list of isosurfaces as ``(vertices,
            faces, band_idx)`` for each spin channel.
        reciprocal_space: The reciprocal space associated with the Fermi surface.
        structure: The structure.
        projections: A property projected onto the surface. The projections are given
            for each face of the Fermi surface. They should be provided as a dict of
            ``{spin: projections}``, where projections is a list of numpy arrays with
            the shape (nfaces, ...), for each surface in ``isosurfaces`. The projections
            can be scalar or vector properties.
        properties: The properties associated with each isosurface. Multiple
            SurfaceProperty objects can be given for a single isosurface, as the
            properties are given for all connected sub-isosurfaces (those that
            are connected together by faces). See the docstring for
            ``isosurface_properties`` for more details.
    """

    isosurfaces: Dict[Spin, List[Tuple[np.ndarray, np.ndarray, int]]]
    reciprocal_space: ReciprocalCell
    structure: Structure
    projections: Optional[Dict[Spin, List[np.ndarray]]] = None
    properties: Optional[Dict[Spin, List[List[SurfaceProperties]]]] = None

    @property
    def n_surfaces(self) -> int:
        """Number of iso-surfaces in the Fermi surface."""
        return sum(map(len, self.isosurfaces.values()))

    @property
    def area(self) -> float:
        """Total area of all iso-surfaces in the Fermi surface."""
        return self.area_surfaces.sum()

    @property
    def area_surfaces(self) -> np.ndarray:
        """Area of each iso-surface in the Fermi surface."""
        from ifermi.analysis import isosurface_area

        areas = [isosurface_area(verts, faces) for verts, faces, _ in self.isosurfaces]
        return np.array(areas)

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
        calculate_properties: bool = False,
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
                given as a dict of ``{spin: projections}``, where projections is numpy
                array with shape (nbands, nkpoints, ...). The number of bands should
                equal the number of bands in the band structure but the k-point mesh
                can be different. Must be used in combination with
                ``kpoints``.
            projection_kpoints: The k-points on which the data is generated.
                Must be used in combination with ``data``.
            calculate_properties: Whether to calculate additional Fermi surface
                properties such as the connectivity and orientation of connected
                sub-surfaces.

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
            calculate_properties=calculate_properties
        )

        properties = None
        if calculate_properties:
            isosurfaces, properties = isosurfaces

        face_projections = None
        if projection_data is not None and projection_kpoints is not None:
            face_projections = get_fermi_surface_projection(
                projection_data,
                projection_kpoints,
                isosurfaces,
                band_structure.structure,
            )
        elif projection_data or projection_kpoints:
            raise ValueError("Both data and kpoints must be specified.")

        return cls(
            isosurfaces, reciprocal_space, structure, face_projections, properties
        )

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

        if fs.projections:
            fs.projections = {Spin(int(k)): v for k, v in fs.projections.items()}

        return fs

    def as_dict(self) -> dict:
        """Get a json-serializable dict representation of FermiSurface."""
        d = super().as_dict()
        d["isosurfaces"] = {str(spin): iso for spin, iso in self.isosurfaces.items()}

        if self.projections:
            d["projections"] = {str(k): v for k, v in self.projections.items()}

        return d


def compute_isosurfaces(
    bands: Dict[Spin, np.ndarray],
    kpoints: np.ndarray,
    fermi_level: float,
    reciprocal_space: ReciprocalCell,
    decimate_factor: Optional[float] = None,
    decimate_method: str = "quadric",
    smooth: bool = False,
    calculate_properties: bool = False,
) -> Union[
     Dict[Spin, List[Tuple[np.ndarray, np.ndarray, int]]],
     Tuple[Dict[Spin, List[Tuple[np.ndarray, np.ndarray, int]]],
           Dict[Spin, List[List[SurfaceProperties]]]]
]:
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
        calculate_properties: Whether to calculate additional Fermi surface
            properties such as the connectivity and orientation of connected
            sub-surfaces.

    Returns:
        A dictionary containing a list of isosurfaces as ``(vertices, faces, band_idx)``
        for each spin channel. If ``calculate_properties``, is True, a list of
        SurfaceProperty objects will be returned for each isosurface in each spin
        channel. See the documentation for ``isosurface_properties`` for more details.
    """
    from skimage.measure import marching_cubes

    from ifermi.analysis import isosurface_properties
    from ifermi.kpoints import get_kpoint_mesh_dim, get_kpoint_spacing

    rlat = reciprocal_space.reciprocal_lattice

    # sort k-points to be in the correct order
    order = np.lexsort((kpoints[:, 2], kpoints[:, 1], kpoints[:, 0]))
    kpoints = kpoints[order]
    bands = {s: b[:, order] for s, b in bands.items()}

    kpoint_dim = get_kpoint_mesh_dim(kpoints)
    spacing = get_kpoint_spacing(kpoints)
    reference = np.min(kpoints, axis=0)

    if smooth and mcubes is None:
        smooth = False
        warnings.warn("Smoothing disabled, install PyMCubes to enable smoothing.")

    if decimate_factor is not None and open3d is None:
        decimate_factor = None
        warnings.warn("Decimation disabled, install open3d to enable decimation.")

    isosurfaces = {}
    properties = {}
    for spin, ebands in bands.items():
        ebands -= fermi_level
        spin_isosurface = []
        spin_properties = []

        for band_idx, band in enumerate(ebands):
            if not np.nanmax(band) > 0 > np.nanmin(band):
                # if band doesn't cross the Fermi level then skip it
                continue

            band_data = band.reshape(kpoint_dim)

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
            verts = np.dot(verts, rlat)

            if calculate_properties:
                spin_properties.append(isosurface_properties(verts, faces, rlat))

            verts, faces = trim_surface(reciprocal_space, verts, faces)
            spin_isosurface.append((verts, faces, and_idx))

        isosurfaces[spin] = spin_isosurface
        properties[spin] = spin_properties

    if calculate_properties:
        return isosurfaces, properties

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


def get_fermi_surface_projection(
    data: Dict[Spin, np.ndarray],
    kpoints: np.ndarray,
    isosurfaces: Dict[Spin, List[Tuple[np.ndarray, np.ndarray, int]]],
    structure: Structure,
) -> Dict[Spin, List[np.ndarray]]:
    """
    Interpolate projections data onto the Fermi surfaces.

    Args:
        data: A property to project onto the Fermi surface. It should be
            given as a dict of ``{spin: projections}``, where projections is numpy
            array with shape (nbands, nkpoints, ...). The number of bands should
            equal the number of bands used to generate the isosurfaces and the number
            of k-points must match ``projection_kpoints``. The projections can be a
            scalar or multidimensional property.
        kpoints: The k-points on which the projection_data is generated.
        isosurfaces: A dictionary containing a list of isosurfaces as ``(vertices,
            faces, band_idx)`` for each spin channel.
        structure: The structure associated with the isosurface.

    Returns:
        The projections data interpolated onto the surface. The projections is given
        for each face of the Fermi surface. The format is a dict of
        ``{spin: projections}``, where projections is a list of numpy arrays with the
        shape (nfaces, ...), for each surface in ``isosurfaces`.
    """
    from ifermi.interpolator import PeriodicLinearInterpolator
    from ifermi.kpoints import kpoints_to_first_bz

    rlat = structure.lattice.reciprocal_lattice
    interpolator = PeriodicLinearInterpolator(kpoints, data)
    projection = {}
    for spin, spin_isosurfaces in isosurfaces.items():
        spin_projections = []
        for vertices, faces, band_idx in spin_isosurfaces:
            # get the center of each of face in cartesian coords
            face_verts = vertices[faces]
            centers = face_verts.mean(axis=1)

            # convert to fractional coords in 1st BZ
            centers = kpoints_to_first_bz(rlat.get_fractional_coords(centers))

            # get interpolated projections at center of faces
            band_idxs = np.full(len(centers), band_idx)
            face_projections = interpolator.interpolate(spin, band_idxs, centers)

            spin_projections.append(face_projections)
        projection[spin] = spin_projections
    return projection


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
