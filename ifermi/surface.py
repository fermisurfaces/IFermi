"""Tools to generate isosurfaces and Fermi surfaces."""

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Tuple, Union

import numpy as np
from monty.dev import requires
from monty.json import MSONable, jsanitize
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.core import Spin

from ifermi.brillouin_zone import ReciprocalCell, WignerSeitzCell
from ifermi.interpolate import LinearInterpolator

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
    "face_properties",
]


@dataclass
class Isosurface(MSONable):
    """
    An isosurface object contains a triangular mesh and surface properties.

    Attributes:
        vertices: A (n, 3) float array of the vertices in the isosurface.
        faces: A (m, 3) int array of the faces of the isosurface.
        band_idx: The band index to which the surface belongs.
        properties: An optional (m, ...) float array containing face properties as
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

    def __post_init__(self):
        # ensure all inputs are numpy arrays

        self.vertices = np.array(self.vertices)
        self.faces = np.array(self.faces)

        if self.properties is not None:
            self.properties = np.array(self.properties)

    @property
    def area(self) -> float:
        r"""Area of the isosurface in Å\ :sup:`-2`\ ."""
        from ifermi.analysis import isosurface_area

        return isosurface_area(self.vertices, self.faces)

    @property
    def has_properties(self) -> float:
        """Whether the surface has properties."""
        return self.properties is not None

    @property
    def properties_norms(self) -> np.ndarray:
        """(m, ) norm of isosurface properties."""
        if not self.has_properties:
            raise ValueError("Isosurface does not have face properties.")

        if self.properties.ndim != 2:
            raise ValueError("Isosurface does not have vector properties.")

        return np.linalg.norm(self.properties, axis=1)

    @property
    def properties_ndim(self) -> int:
        """Dimensionality of face properties."""
        if not self.has_properties:
            raise ValueError("Isosurface does not have face properties.")

        return self.properties.ndim

    def average_properties(
        self, norm: bool = False, projection_axis: Optional[Tuple[int, int, int]] = None
    ) -> Union[float, np.ndarray]:
        """
        Average property across isosurface.

        Args:
            norm: Average the norm of the properties (vector properties only).
            projection_axis: A (3, ) in array of the axis to project the properties onto
                (vector properties only).

        Returns:
            The averaged property.
        """
        from ifermi.analysis import average_properties

        if not self.has_properties:
            raise ValueError("Isosurface does not have face properties.")

        properties = self.properties
        if projection_axis is not None:
            properties = self.scalar_projection(projection_axis)

        return average_properties(self.vertices, self.faces, properties, norm=norm)

    def scalar_projection(self, axis: Tuple[int, int, int]) -> np.ndarray:
        """
        Get scalar projection of properties onto axis.

        Args:
            axis: A (3, ) int array of the axis to project onto.

        Return:
            (m, ) float array of scalar projection of properties.
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

    def __repr__(self):
        rep = [
            f"Isosurface(nvertices={len(self.vertices)}, "
            f"nfaces={len(self.faces)}, band_idx={self.band_idx}",
        ]
        if self.dimensionality is not None:
            rep.append(f", dim={self.dimensionality}")
        if self.orientation is not None:
            rep.append(f", orientation={self.orientation}")
        rep.append(")")
        return "".join(rep)


@dataclass
class FermiSurface(MSONable):
    """
    A FermiSurface object contains isosurfaces and the reciprocal lattice definition.

    Attributes:
        isosurfaces: A dict containing a list of isosurfaces for each spin channel.
        reciprocal_space: A reciprocal space defining periodic boundary conditions.
        structure: The structure.
    """

    isosurfaces: Dict[Spin, List[Isosurface]]
    reciprocal_space: ReciprocalCell
    structure: Structure

    @property
    def n_surfaces(self) -> int:
        """Number of isosurfaces in the Fermi surface."""
        return sum(self.n_surfaces_per_spin.values())

    @property
    def n_surfaces_per_band(self) -> Dict[Spin, Dict[int, int]]:
        """
        Get number of surfaces for each band index for each spin channel.

        Returned as a dict of ``{spin: {band_idx: count}}``.
        """
        from collections import Counter

        n_surfaces = {}
        for spin, isosurfaces in self.isosurfaces.items():
            n_surfaces[spin] = dict(Counter([s.band_idx for s in isosurfaces]))

        return n_surfaces

    @property
    def n_surfaces_per_spin(self) -> Dict[Spin, int]:
        """
        Get number of surfaces per spin channel.

        Returned as a dict of ``{spin: count}``.
        """
        return {spin: len(surfaces) for spin, surfaces in self.isosurfaces.items()}

    @property
    def area(self) -> float:
        r"""Total area of all isosurfaces in the Fermi surface in Å\ :sup:`-2`\ ."""
        return sum(map(sum, self.area_surfaces.values()))

    @property
    def area_surfaces(self) -> Dict[Spin, np.ndarray]:
        r"""Area of each isosurface in the Fermi surface in Å\ :sup:`-2`\ ."""
        return {k: np.array([i.area for i in v]) for k, v in self.isosurfaces.items()}

    @property
    def has_properties(self) -> bool:
        """Whether all isosurfaces have face properties."""
        return all(
            [all([i.has_properties for i in s]) for s in self.isosurfaces.values()]
        )

    def average_properties(
        self, norm: bool = False, projection_axis: Optional[Tuple[int, int, int]] = None
    ) -> Union[float, np.ndarray]:
        """
        Average property across the full Fermi surface.

        Args:
            norm: Average the norm of the properties (vector properties only).
            projection_axis: A (3, ) in array of the axis to project the properties onto
                (vector properties only).

        Returns:
            The averaged property.
        """
        surface_averages = self.average_properties_surfaces(norm, projection_axis)
        surface_areas = self.area_surfaces

        scaled_average = 0
        total_area = 0
        for spin in self.spins:
            for average, area in zip(surface_averages[spin], surface_areas[spin]):
                scaled_average += average * area
                total_area += area

        return scaled_average / total_area

    def average_properties_surfaces(
        self, norm: bool = False, projection_axis: Optional[Tuple[int, int, int]] = None
    ) -> Dict[Spin, List[Union[float, np.ndarray]]]:
        """
        Average property for each isosurface in the Fermi surface.

        Args:
            norm: Average the norm of the properties (vector properties only).
            projection_axis: A (3, ) in array of the axis to project the properties onto
                (vector properties only).

        Returns:
            The averaged property for each surface in each spin channel.
        """
        return {
            k: [i.average_properties(norm, projection_axis) for i in v]
            for k, v in self.isosurfaces.items()
        }

    @property
    def properties_ndim(self) -> int:
        """Dimensionality of isosurface properties."""
        if not self.has_properties:
            raise ValueError("Isosurfaces don't have properties.")

        ndims = [i.properties_ndim for v in self.isosurfaces.values() for i in v]

        if len(set(ndims)) != 1:
            warnings.warn(
                "Isosurfaces have different property dimensions, using the largest."
            )

        return max(ndims)

    @property
    def spins(self):
        """The spin channels in the Fermi surface."""
        return tuple(self.isosurfaces.keys())

    def all_vertices_faces(
        self, spins: Optional[Union[Spin, Collection[Spin]]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the vertices and faces for all isosurfaces.

        Args:
            spins: One or more spin channels to select. Default is all spins available.

        Returns:
            A list of (vertices, faces).
        """
        if not spins:
            spins = self.spins
        elif isinstance(spins, Spin):
            spins = [spins]

        vertices_faces = []
        for spin in spins:
            for isosurface in self.isosurfaces[spin]:
                vertices_faces.append((isosurface.vertices, isosurface.faces))

        return vertices_faces

    def all_properties(
        self,
        spins: Optional[Union[Spin, Collection[Spin]]] = None,
        projection_axis: Optional[Tuple[int, int, int]] = None,
        norm: bool = False,
    ) -> List[np.ndarray]:
        """
        Get the properties for all isosurfaces.

        Args:
            spins: One or more spin channels to select. Default is all spins available.
            projection_axis: A (3, ) in array of the axis to project the properties onto
                (vector properties only).
            norm: Calculate the norm of the properties (vector properties only).
                Ignored if ``projection_axis`` is set.

        Returns:
            A list of properties arrays for each isosurface.
        """
        if not spins:
            spins = self.spins
        elif isinstance(spins, Spin):
            spins = [spins]

        projections = []
        for spin in spins:
            for isosurface in self.isosurfaces[spin]:
                if projection_axis is not None:
                    projections.append(isosurface.scalar_projection(projection_axis))
                elif norm:
                    projections.append(isosurface.properties_norms)
                else:
                    projections.append(isosurface.properties)

        return projections

    @classmethod
    def from_band_structure(
        cls,
        band_structure: BandStructure,
        mu: float = 0.0,
        wigner_seitz: bool = False,
        decimate_factor: Optional[float] = None,
        decimate_method: str = "quadric",
        smooth: bool = False,
        property_data: Optional[Dict[Spin, np.ndarray]] = None,
        property_kpoints: Optional[np.ndarray] = None,
        calculate_dimensionality: bool = False,
    ) -> "FermiSurface":
        """
        Create a FermiSurface from a pymatgen band structure object.

        Args:
            band_structure: A band structure. The k-points must cover the full
                Brillouin zone (i.e., not just be the irreducible mesh). Use
                the ``ifermi.interpolator.FourierInterpolator`` class to expand the k-points to
                the full Brillouin zone if required.
            mu: Energy offset from the Fermi energy at which the isosurface is
                calculated.
            wigner_seitz: Controls whether the cell is the Wigner-Seitz cell
                or the reciprocal unit cell parallelepiped.
            decimate_factor: If method is "quadric", and factor is a floating point
                value then factor is the scaling factor by which to reduce the number of
                faces. I.e., final # faces = initial # faces * factor. If method is
                "quadric" but factor is an integer then factor is the target number of
                final faces. If method is "cluster", factor is the voxel size in which
                to cluster points. Default is None (no decimation).
            decimate_method: Algorithm to use for decimation. Options are "quadric"
                or "cluster".
            smooth: If True, will smooth resulting isosurface. Requires PyMCubes. See
                compute_isosurfaces for more information.
            property_data: A property to project onto the Fermi surface. It should be
                given as a dict of ``{spin: properties}``, where properties is numpy
                array with shape (nbands, nkpoints, ...). The number of bands should
                equal the number of bands in the band structure but the k-point mesh
                can be different. Must be used in combination with
                ``kpoints``.
            property_kpoints: The k-points on which the data is generated.
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
        if property_data is not None and property_kpoints is not None:
            interpolator = LinearInterpolator(property_kpoints, property_data)
        elif property_data is not None or property_kpoints is not None:
            raise ValueError("Both data and kpoints must be specified.")

        bands, kpoints = expand_bands(
            bands, kpoints, supercell_dim=(3, 3, 3), center=(1, 1, 1)
        )
        if isinstance(decimate_factor, int):
            # increase number of target faces to account for 3x3x3 supercell
            decimate_factor *= 27

        isosurfaces = compute_isosurfaces(
            bands,
            kpoints,
            fermi_level,
            reciprocal_space,
            decimate_factor=decimate_factor,
            decimate_method=decimate_method,
            smooth=smooth,
            calculate_dimensionality=calculate_dimensionality,
            property_interpolator=interpolator,
        )

        return cls(isosurfaces, reciprocal_space, structure)

    def get_fermi_slice(
        self, plane_normal: Tuple[int, int, int], distance: float = 0
    ) -> "FermiSlice":
        """
        Get a slice through the Fermi surface.

        Slice defined by the intersection of a plane with the Fermi surface.

        Args:
            plane_normal: (3, ) int array of the plane normal in fractional indices.
            distance: The distance from the center of the Brillouin zone (Γ-point).

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
        return jsanitize(d, strict=True)


def compute_isosurfaces(
    bands: Dict[Spin, np.ndarray],
    kpoints: np.ndarray,
    fermi_level: float,
    reciprocal_space: ReciprocalCell,
    decimate_factor: Optional[float] = None,
    decimate_method: str = "quadric",
    smooth: bool = False,
    calculate_dimensionality: bool = False,
    property_interpolator: Optional[LinearInterpolator] = None,
) -> Dict[Spin, List[Isosurface]]:
    """
    Compute the isosurfaces at a particular energy level.

    Args:
        bands: The band energies, given as a dictionary of ``{spin: energies}``, where
            energies has the shape (nbands, nkpoints).
        kpoints: The k-points in fractional coordinates.
        fermi_level: The energy at which to calculate the Fermi surface.
        reciprocal_space: The reciprocal space representation.
        decimate_factor: If method is "quadric", and factor is a floating point value
            then factor is the scaling factor by which to reduce the number of faces.
            I.e., final # faces = initial # faces * factor. If method is "quadric" but
            factor is an integer then factor is the target number of final faces.
            If method is "cluster", factor is the voxel size in which to cluster points.
            Default is None (no decimation).
        decimate_method: Algorithm to use for decimation. Options are "quadric" or
            "cluster".
        smooth: If True, will smooth resulting isosurface. Requires PyMCubes. Smoothing
            algorithm will use constrained smoothing algorithm to preserve fine details
            if input dimension is lower than (500, 500, 500), otherwise will apply a
            Gaussian filter.
        calculate_dimensionality: Whether to calculate isosurface dimensionality.
        property_interpolator: An interpolator class for interpolating properties
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
                property_interpolator,
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
    property_interpolator: Optional[LinearInterpolator],
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
    properties = None
    for (subverts, subfaces), idx in zip(subsurfaces, mapping):
        # convert vertices to cartesian coordinates
        subverts = np.dot(subverts, rlat)
        subverts, subfaces = trim_surface(reciprocal_space, subverts, subfaces)

        if len(subverts) == 0:
            # skip surfaces that do not enter the reciprocal space boundaries
            continue

        if calculate_dimensionality:
            dimensionality, orientation = dimensionalities[idx]

        if property_interpolator is not None:
            properties = face_properties(
                property_interpolator, subverts, subfaces, band_idx, spin, rlat
            )

        isosurface = Isosurface(
            vertices=subverts,
            faces=subfaces,
            band_idx=band_idx,
            properties=properties,
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
        vertices: A (n, 3) float array of the vertices in the isosurface.
        faces: A (m, 3) int array of the faces of the isosurface.

    Returns:
        (vertices, faces) of the trimmed surface.
    """
    from trimesh.intersections import slice_faces_plane

    for center, normal in zip(reciprocal_cell.centers, reciprocal_cell.normals):
        vertices, faces = slice_faces_plane(vertices, faces, -normal, center)
    return vertices, faces


def expand_bands(
    bands: Dict[Spin, np.ndarray],
    fractional_kpoints: np.ndarray,
    supercell_dim: Tuple[int, int, int] = (3, 3, 3),
    center: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Dict[Spin, np.ndarray], np.ndarray]:
    """
    Expand the band energies and k-points with periodic boundary conditions.

    Args:
        bands: The band energies, given as a dictionary of ``{spin: energies}``, where
            energies has the shape (nbands, nkpoints).
        fractional_kpoints: A (n, 3) float array of the fractional k-point coordinates.
        supercell_dim: The supercell mesh dimensions.
        center: The cell on which the supercell is centered.

    Returns:
        (energies, kpoints) The expanded band energies and k-points.
    """
    final_ebands = {}
    nk = len(fractional_kpoints)
    ncells = np.product(supercell_dim)

    final_kpoints = np.tile(fractional_kpoints, (ncells, 1))
    for n, (i, j, k) in enumerate(np.ndindex(supercell_dim)):
        final_kpoints[n * nk : (n + 1) * nk] += [i, j, k]
    final_kpoints -= center

    for spin, ebands in bands.items():
        final_ebands[spin] = np.tile(ebands, (1, ncells))

    return final_ebands, final_kpoints


def face_properties(
    interpolator: LinearInterpolator,
    vertices: np.ndarray,
    faces: np.ndarray,
    band_idx: int,
    spin: Spin,
    reciprocal_lattice: np.ndarray,
) -> np.ndarray:
    """
    Interpolate properties data onto the Fermi surfaces.

    Args:
        interpolator: Periodic interpolator for property data.
        vertices: A (n, 3) float array of the vertices in the isosurface.
        faces: A (m, 3) int array of the faces of the isosurface.
        band_idx: The band that the isosurface belongs to.
        spin: The spin channel the isosurface belongs to.
        reciprocal_lattice: Reciprocal lattice matrix

    Returns:
        A (m, ...) float array of the interpolated properties at the center of each
        face in the isosurface.
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
def decimate_mesh(
    vertices: np.ndarray, faces: np.ndarray, factor: Union[int, float], method="quadric"
):
    """Decimate mesh to reduce the number of triangles and vertices.

    The open3d package is required for decimation.

    Args:
        vertices: A (n, 3) float array of the vertices in the isosurface.
        faces: A (m, 3) int array of the faces of the isosurface.
        factor: If method is "quadric", and factor is a floating point value then
            factor is the scaling factor by which to reduce the number of faces.
            I.e., final # faces = initial # faces * factor. If method is "quadric" but
            factor is an integer then factor is the target number of final faces.
            If method is "cluster", factor is the voxel size in which to cluster points.
        method: Algorithm to use for decimation. Options are "quadric" or "cluster".

    Returns:
        (vertices, faces) of the decimated mesh.
    """
    # convert mesh to open3d format
    o3d_verts = open3d.utility.Vector3dVector(vertices)
    o3d_faces = open3d.utility.Vector3iVector(faces)
    o3d_mesh = open3d.geometry.TriangleMesh(o3d_verts, o3d_faces)

    # decimate mesh
    if method == "quadric":
        if isinstance(factor, int):
            n_target_triangles = min(factor, len(faces))
        else:
            n_target_triangles = int(len(faces) * factor)
        o3d_new_mesh = o3d_mesh.simplify_quadric_decimation(n_target_triangles)
    else:
        cluster_type = open3d.geometry.SimplificationContraction.Quadric
        o3d_new_mesh = o3d_mesh.simplify_vertex_clustering(factor, cluster_type)

    return np.array(o3d_new_mesh.vertices), np.array(o3d_new_mesh.triangles)
