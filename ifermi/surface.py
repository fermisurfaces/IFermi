"""
This module contains the classes and methods for creating iso-surface structures
from Pymatgen structure objects. The iso-surfaces are found using the
Scikit-image package.
"""
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from monty.dev import requires
from monty.json import MSONable
from pymatgen import Spin, Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from skimage.measure import marching_cubes
from trimesh import Trimesh
from trimesh.intersections import mesh_multiplane, slice_faces_plane

from ifermi.brillouin_zone import ReciprocalCell, ReciprocalSlice, WignerSeitzCell
from ifermi.interpolator import PeriodicLinearInterpolator
from ifermi.kpoints import (
    get_kpoint_mesh_dim,
    get_kpoint_spacing,
    get_kpoints_from_bandstructure,
    kpoints_to_first_bz,
)

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
            ``{spin: (spin_slices, band_idx)}`` where spin_slices is a List of numpy
            arrays, each with the shape ``(n_lines, 2, 2)``.
        reciprocal_slice: The reciprocal slice defining the intersection of the
            plane with the Brillouin zone edges.
        structure: The structure.
        projections: A property projected onto the slice. The projections are given
            for each line. They should be provided as a dict of
            ``{spin: projections}``, where projections is a list of numpy arrays with
            the shape (n_lines, ...), for each slice in ``slices`. The projections
            can scalar or vector properties.

    """

    slices: Dict[Spin, List[Tuple[np.ndarray, int]]]
    reciprocal_slice: ReciprocalSlice
    structure: Structure
    projections: Optional[Dict[Spin, List[np.ndarray]]] = None

    @classmethod
    def from_dict(cls, d) -> "FermiSlice":
        """Returns FermiSurface object from dict."""
        fs = super().from_dict(d)
        fs.slices = {Spin(int(k)): v for k, v in fs.slices.items()}

        if fs.projections:
            fs.projections = {Spin(int(k)): v for k, v in fs.projections.items()}

        return fs

    def as_dict(self) -> dict:
        """Get a json-serializable dict representation of FermiSurface."""
        d = super().as_dict()
        d["slices"] = {str(spin): iso for spin, iso in self.slices.items()}

        if self.projections:
            d["projections"] = {str(k): v for k, v in self.projections.items()}

        return d


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
            can scalar or vector properties.

    """

    isosurfaces: Dict[Spin, List[Tuple[np.ndarray, np.ndarray, int]]]
    reciprocal_space: ReciprocalCell
    structure: Structure
    projections: Optional[Dict[Spin, List[np.ndarray]]] = None

    @property
    def n_surfaces(self) -> int:
        return sum(map(len, self.isosurfaces.values()))

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
    ) -> "FermiSurface":
        """
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
        """
        band_structure = deepcopy(band_structure)  # prevent data getting overwritten

        structure = band_structure.structure
        fermi_level = band_structure.efermi + mu
        bands = band_structure.bands
        kpoints = get_kpoints_from_bandstructure(band_structure)

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
        )

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

        return cls(isosurfaces, reciprocal_space, structure, face_projections)

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
        projections = {}
        for spin, spin_isosurfaces in self.isosurfaces.items():
            spin_slices = []
            spin_projections = []

            for i, (verts, faces, band_idx) in enumerate(spin_isosurfaces):
                mesh = Trimesh(vertices=verts, faces=faces)
                lines, _, face_idxs = mesh_multiplane(mesh, cart_origin, cart_normal, [0])

                # only provided one mesh, so get the segments and faces for that
                segments = lines[0]
                face_idxs = face_idxs[0]

                if len(segments) == 0:
                    # plane did not intersect surface
                    continue

                paths = process_lines(segments, face_idxs)

                for path_segments, path_faces in zip(paths):
                    spin_slices.append((path_segments, band_idx))
                    if self.projections:
                        spin_projections.append(self.projections[spin][i][path_faces])

            slices[spin] = spin_slices
            if self.projections:
                projections[spin] = spin_projections
            else:
                projections = None

        reciprocal_slice = self.reciprocal_space.get_reciprocal_slice(
            plane_normal, distance
        )

        return FermiSlice(slices, reciprocal_slice, self.structure, projections)

    @classmethod
    def from_dict(cls, d) -> "FermiSurface":
        """Returns FermiSurface object from dict."""
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
) -> Dict[Spin, List[Tuple[np.ndarray, np.ndarray, int]]]:
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

    Returns:
        A dictionary containing a list of isosurfaces as ``(vertices, faces, band_idx)``
        for each spin channel.
    """
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
    for spin, ebands in bands.items():
        ebands -= fermi_level
        spin_isosurface = []

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
            verts, faces = trim_surface(reciprocal_space, verts, faces)

            spin_isosurface.append((verts, faces, band_idx))

        isosurfaces[spin] = spin_isosurface

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


def process_lines(
    segments: np.ndarray, face_idxs: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Process segments and face_idxs from mesh_multiplane.

    The key issue is that the segments from mesh_multiplane do not correspond to
    individual lines, nor are they sorted in a continuous order. Instead they are just
    a list of randomly ordered segments. This causes trouble later on when trying
    to add equally spaced arrows to the lines.

    The goal of this function is to identify the separate paths in the segments (a path
    is a collection of segments that are connected together), and return these
    segments in the order that they are connected. By looping through each segment, you
    will be proceeding along the line in a certain direction.

    Because the original segments may contain multiple paths, a list of segments
    and their corresponding face indices are returned.

    Lastly, note that there is no guarantee that all of the original segments will be
    used in the processed segments. This is because some of the original segments are
    very small and will be filtered out.

    Args:
        segments: The segments from mesh_multiplane as an array with the shape
            (nsegments, 2, 2).
        face_idxs: The face indices that each segment belongs to.

    Returns:
        A list of segments and faces for each path.
    """

    # turn the segments with the shape (nsegments, 2, 2), into a list of vertices
    # with the shape (nsegments * 2, 2)
    vertices = segments.reshape(-1, 2)

    # create edges that correspond to the original segments, i.e., [(0, 1), (2, 3), ...]
    edges = np.arange(0, len(vertices)).reshape(len(segments), 2)

    # merge vertices that are close together and get an equivalence mapping
    mapping = get_equivalent_vertices(vertices)

    # get the indices of the unique vertices
    unique_vertices_idx = np.unique(mapping)

    # use only equivalent vertices for the edges
    edges = mapping[edges]

    # some of the edges may now be duplicates, and some even may be edges between
    # the same vertex; filter these duplicates/self edges but keep track of the index
    # of the unique edges in the original edges array. We keep this information as this
    # index is used to identify the face that each segment belongs to
    unique_edges, unique_edge_idxs = np.unique(edges, axis=0, return_index=True)

    # filter self edges
    non_self_edges = unique_edges[:, 0] != unique_edges[:, 1]

    # these are the unique/non-self edges
    unique_edges = unique_edges[non_self_edges]

    # these are the indices of the unique edges in the original edges array
    unique_edge_idxs = unique_edge_idxs[non_self_edges]

    # create a mapping from the edge data to the original edge index
    edge_mapping = {(u, v): idx for (u, v), idx in zip(unique_edges, unique_edge_idxs)}

    # get the longest paths for each subgraph
    paths = get_longest_simple_paths(unique_vertices_idx, unique_edges)

    # get the new segments and corresponding face indices for each path
    path_data = []
    for path in paths:
        pair_path = np.array(list(_pairwise(path)))
        new_segments = vertices[pair_path]

        edge_idxs = np.array([edge_mapping[(u, v)] for u, v in pair_path])
        new_faces = face_idxs[edge_idxs]

        path_data.append((new_segments, new_faces))

    return path_data


def get_equivalent_vertices(vertices: np.ndarray, tol: float = 1e-4):
    from scipy import spatial

    tree = spatial.cKDTree(vertices)
    to_merge = tree.query_ball_tree(tree, tol)

    merge_mapping = {}
    seen = set()
    for i, merge_set in enumerate(to_merge):
        if i in merge_mapping or i in seen:
            continue

        merge_mapping[i] = {i}
        seen.add(i)
        queue = list(merge_set)
        for idx in queue:
            if idx in seen:
                continue

            merge_mapping[i].add(idx)
            seen.add(idx)
            queue += list(to_merge[i])

    inverse_mapping = {}
    for k, v in merge_mapping.items():
        inverse_mapping.update(dict(zip(list(v), len(v) * [k])))

    return np.array([inverse_mapping[i] for i in range(len(vertices))])


def get_longest_simple_paths(vertices, edges):
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)

    subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]

    paths = []
    for subgraph in subgraphs:
        cycles = list(nx.cycle_basis(subgraph))

        if len(cycles) > 0:
            # path contains a cycle, i.e., the path does not hit a periodic boundary
            # there should be only one cycle, but just in case we take the longest one
            longest_path = sorted(cycles, key=lambda x: len(x))[-1]

        else:
            # graph does not have cycles, i.e., it his the periodic boundary condition
            # twice; find the nodes with only one edge and use these as the start
            # and end points for the path
            ends = [v for v, d in subgraph.degree if d == 1]
            if len(ends) != 2:
                raise ValueError("Path is not unique; valued to find path")

            start, end = ends
            simple_paths = list(nx.all_simple_paths(subgraph, start, end))
            longest_path = sorted(simple_paths, key=lambda x: len(x))[-1]

        if len(longest_path) != len(graph.nodes):
            raise ValueError("Path does not cover all nodes.")

        paths.append(longest_path)

    return paths


def _pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)