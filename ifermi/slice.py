"""
This module contains tools for creating Fermi slices from FermiSurface objects.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from monty.json import MSONable

from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin

from ifermi.brillouin_zone import ReciprocalSlice

__all__ = [
    "FermiSlice",
    "process_lines",
    "get_equivalent_vertices",
    "get_longest_simple_paths",
    "interpolate_segments",
]


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
    def from_fermi_surface(
        cls,
        fermi_surface: "FermiSurface",
        plane_normal: Tuple[int, int, int],
        distance: float = 0,
    ) -> "FermiSlice":
        """
        Get a slice through the Fermi surface, defined by the intersection of a plane
        with the fermi surface.

        Args:
            fermi_surface: A Fermi surface.
            plane_normal: The plane normal in fractional indices. E.g., ``(1, 0, 0)``.
            distance: The distance from the center of the Brillouin zone (the Gamma
                point).

        Returns:
            The Fermi slice.

        """
        from trimesh import Trimesh
        from trimesh.intersections import mesh_multiplane

        cart_normal = np.dot(
            plane_normal, fermi_surface.reciprocal_space.reciprocal_lattice
        )
        cart_origin = cart_normal * distance

        slices = {}
        projections = {}
        for spin, spin_isosurfaces in fermi_surface.isosurfaces.items():
            spin_slices = []
            spin_projections = []

            for i, (verts, faces, band_idx) in enumerate(spin_isosurfaces):
                mesh = Trimesh(vertices=verts, faces=faces)
                lines, _, face_idxs = mesh_multiplane(
                    mesh, cart_origin, cart_normal, [0]
                )

                # only provided one mesh, so get the segments and faces for that
                segments = lines[0]
                face_idxs = face_idxs[0]

                if len(segments) == 0:
                    # plane did not intersect surface
                    continue

                paths = process_lines(segments, face_idxs)

                for path_segments, path_faces in paths:
                    path_projections = fermi_surface.projections[spin][i][path_faces]
                    path_segments, path_projections = interpolate_segments(
                        path_segments, path_projections, 0.001
                    )
                    spin_slices.append((path_segments, band_idx))
                    if fermi_surface.projections:
                        spin_projections.append(path_projections)

            slices[spin] = spin_slices
            if fermi_surface.projections:
                projections[spin] = spin_projections
            else:
                projections = None

        reciprocal_slice = fermi_surface.reciprocal_space.get_reciprocal_slice(
            plane_normal, distance
        )

        return FermiSlice(
            slices, reciprocal_slice, fermi_surface.structure, projections
        )

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
    edge_mapping = {
        (min(u, v), max(u, v)): idx
        for (u, v), idx in zip(unique_edges, unique_edge_idxs)
    }

    # get the longest paths for each subgraph
    paths = get_longest_simple_paths(unique_vertices_idx, unique_edges)

    # get the new segments and corresponding face indices for each path
    path_data = []
    for path in paths:
        pair_path = np.array(list(_pairwise(path)))
        new_segments = vertices[pair_path]

        edge_idxs = np.array(
            [edge_mapping[(min(u, v), max(u, v))] for u, v in pair_path]
        )
        new_faces = face_idxs[edge_idxs]

        path_data.append((new_segments, new_faces))

    return path_data


def get_equivalent_vertices(vertices: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """
    Finds vertices that are equivalent (closer than a tolerance).

    Note that the algorithm used is effectively recursive. If vertex a is within the
    tolerance of b, and b is within the tolerance of c, even if a and c and not within
    the tolerance all vertices will be equivalent.

    Args:
        vertices: The vertices as a numpy array with shape (nvertices, 2).
        tol: The distance tolerance for equivalence.

    Returns:
        The mapping that maps each vertex in the original vertex array to its
        equivalent index.
    """
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


def get_longest_simple_paths(
    vertices: np.ndarray, edges: np.ndarray
) -> List[np.ndarray]:
    """
    Find the shortest paths that go through all nodes.

    The paths are broken up into the connected subgraphs. Note this function is only
    designed to work with connected subgraphs that are either simple cycles or
    chains with no branches.

    Args:
        vertices: The vertices as a numpy array with shape (nvertices, 2).
        edges: The edges as a numpy array with the shape (nedges, 2).

    Returns:
        A list of paths (one for each connected subpath), specified as an array of ints.
    """
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
            longest_path.append(longest_path[0])  # complete the cycle

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

        if set(longest_path) != set(subgraph.nodes):
            raise ValueError("Path does not cover all nodes.")

        paths.append(longest_path)

    return paths


def interpolate_segments(
    segments: np.ndarray,  projections: np.ndarray, max_spacing: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a series of line segments to a consistent density.

    Note: the segments must be ordered so that they are adjacent.

    Args:
        segments: The line segments as a numpy array with the shape (nsegments, 2, 2).
        projections: The line projections as an array with the shape (nsegments, ).
        max_spacing: The desired spacing after interpolation. Note, the spacing
            may be slightly smaller than this value.

    Returns:
        The interpolated segments and projections.
    """
    from scipy.interpolate import interp1d

    is_cycle = np.allclose(segments[0, 0], segments[-1, 1], atol=1e-4)

    if len(segments) < 3:
        return segments, projections

    vert = np.concatenate([segments[:, 0], segments[-1, 1][None]])
    lengths = np.linalg.norm(vert[:-1] - vert[1:], axis=1)
    length = np.sum(lengths)

    vert_dist = np.concatenate([[0], np.cumsum(lengths)])
    proj_dist = np.concatenate([[0], (vert_dist[:-1] + vert_dist[1:]) / 2, [length]])

    if is_cycle:
        proj_start = [(projections[0] + projections[-1]) / 2]
        projections = np.concatenate([proj_start, projections, proj_start])
    else:
        projections = np.concatenate([projections[0], projections, projections[-1]])

    vert_interpolator = interp1d(
        vert_dist,
        vert,
        kind="quadratic",
        axis=0,
        bounds_error=False,
        fill_value="extrapolate"
    )
    proj_interpolator = interp1d(
        proj_dist,
        projections,
        kind="linear",
        axis=0,
        bounds_error=False,
        fill_value="extrapolate"
    )

    vert_xs = np.linspace(0, length, int(np.ceil(length / max_spacing)))
    proj_xs = (vert_xs[:-1] + vert_xs[1:]) / 2

    new_vert = vert_interpolator(vert_xs)
    new_proj = proj_interpolator(proj_xs)

    new_segments = np.array(list(_pairwise(new_vert)))
    return new_segments, new_proj


def _pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    from itertools import tee

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
