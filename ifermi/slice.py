"""Tools for creating Fermi isolines from FermiSurface objects."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin

from ifermi.analysis import equivalent_vertices, longest_simple_paths
from ifermi.brillouin_zone import ReciprocalSlice

__all__ = [
    "FermiSlice",
    "process_lines",
    "interpolate_segments",
]


@dataclass
class Isoline(MSONable):
    """
    An isoline object contains line segments mesh and line properties.

    Attributes:
        segments: A (n, 2, 2) float array of the line segments..
        band_idx: The band index to which the slice belongs.
        properties: An optional (n, ...) float array containing segment properties as
            scalars or vectors.
    """
    segments: np.ndarray
    band_idx: int
    properties: Optional[np.ndarray] = None

    @property
    def has_properties(self) -> float:
        """Whether the isoline has properties."""
        return self.properties is None

    def scalar_projection(self, axis: Tuple[int, int, int]) -> np.ndarray:
        """
        Get scalar projection of properties onto axis.

        Args:
            axis: A (3, ) int array of the axis to project onto.
        """
        if not self.has_properties:
            raise ValueError("Isoline does not have face properties.")

        if self.properties.ndim != 2:
            raise ValueError("Isoline does not have vector properties.")

        return np.dot(self.properties, axis)


@dataclass
class FermiSlice(MSONable):
    """
    A FermiSlice object is a 2D slice through a Fermi surface.

    Attributes:
        isolines: A dict containing a list of isolines for each spin channel.
        reciprocal_space: A reciprocal slice defining the intersection of the slice
            with the Brillouin zone edges.
        structure: The structure.
    """

    isolines: Dict[Spin, List[Isoline]]
    reciprocal_slice: ReciprocalSlice
    structure: Structure

    @property
    def n_lines(self) -> int:
        """Number of isolines in the Fermi surface."""
        return sum(map(len, self.isolines.values()))

    @property
    def has_properties(self) -> bool:
        """Whether all isolines have segment properties."""
        return all(
            [all([i.has_properties for i in s]) for s in self.isolines.values()]
        )

    @property
    def spins(self) -> Tuple[Spin]:
        """The spin channels in the Fermi slice."""
        return tuple(self.isolines.keys())

    @classmethod
    def from_fermi_surface(
        cls,
        fermi_surface: "FermiSurface",
        plane_normal: Tuple[int, int, int],
        distance: float = 0,
    ) -> "FermiSlice":
        """Get a slice through the Fermi surface.

        The slice is defined by the intersection of a plane with the Fermi surface.

        Args:
            fermi_surface: A Fermi surface object.
            plane_normal: (3, ) int array of the plane normal in fractional indices.
            distance: The distance from the center of the Brillouin zone (Γ-point).

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
        properties = {}
        for spin, spin_isosurfaces in fermi_surface.isosurfaces.items():
            spin_slices = []
            spin_properties = []

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
                    path_properties = fermi_surface.properties[spin][i][path_faces]
                    path_segments, path_properties = interpolate_segments(
                        path_segments, path_properties, 0.001
                    )
                    spin_slices.append((path_segments, band_idx))
                    if fermi_surface.properties:
                        spin_properties.append(path_properties)

            slices[spin] = spin_slices
            if fermi_surface.properties:
                properties[spin] = spin_properties
            else:
                properties = None

        reciprocal_slice = fermi_surface.reciprocal_space.get_reciprocal_slice(
            plane_normal, distance
        )

        return FermiSlice(
            slices, reciprocal_slice, fermi_surface.structure, properties
        )

    @classmethod
    def from_dict(cls, d) -> "FermiSlice":
        """Return FermiSlice object from a dict."""
        fs = super().from_dict(d)
        fs.isolines = {Spin(int(k)): v for k, v in fs.isolines.items()}
        return fs

    def as_dict(self) -> dict:
        """Get a json-serializable dict representation of a FermiSlice."""
        d = super().as_dict()
        d["isolines"] = {str(spin): iso for spin, iso in self.isolines.items()}
        return d


def process_lines(
    segments: np.ndarray, face_idxs: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Process segments and face_idxs from mesh_multiplane.

    The key issue is that the segments from mesh_multiplane do not correspond to
    individual lines, nor are they sorted in a continuous order. Instead they are just
    a list of randomly ordered segments. This causes trouble later on when trying
    to interpolate the lines or add equally spaced arrows.

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
        segments: A (n, 2, 2) float array of the line segments.
        face_idxs: The face indices that each segment belongs to.

    Returns:
        A list of (segments, faces) for each path.
    """

    # turn segments [shape: (nsegments, 2, 2)], to vertices [shape: (nsegments * 2, 2)]
    vertices = segments.reshape(-1, 2)

    # create edges that correspond to the original segments, i.e., [(0, 1), (2, 3), ...]
    edges = np.arange(0, len(vertices)).reshape(len(segments), 2)

    # merge vertices that are close together and get an equivalence mapping
    mapping = equivalent_vertices(vertices)

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
    paths = longest_simple_paths(unique_vertices_idx, unique_edges)

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


def interpolate_segments(
    segments: np.ndarray, properties: np.ndarray, max_spacing: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a series of line segments to a consistent density.

    Note: the segments must be ordered so that they are adjacent.

    Args:
        segments: A (n, 2, 2) float array of the line segments.
        properties: A (n, ...) float array of the segment properties.
        max_spacing: The desired spacing after interpolation. Note, the spacing
            may be slightly smaller than this value.

    Returns:
        (segments, properties): The interpolated segments and properties.
    """
    from scipy.interpolate import interp1d

    is_cycle = np.allclose(segments[0, 0], segments[-1, 1], atol=1e-4)

    if len(segments) < 3:
        return segments, properties

    vert = np.concatenate([segments[:, 0], segments[-1, 1][None]])
    lengths = np.linalg.norm(vert[:-1] - vert[1:], axis=1)
    length = np.sum(lengths)

    vert_dist = np.concatenate([[0], np.cumsum(lengths)])
    proj_dist = np.concatenate([[0], (vert_dist[:-1] + vert_dist[1:]) / 2, [length]])

    if is_cycle:
        proj_start = [(properties[0] + properties[-1]) / 2]
        properties = np.concatenate([proj_start, properties, proj_start])
    else:
        properties = np.concatenate([properties[0], properties, properties[-1]])

    vert_interpolator = interp1d(
        vert_dist,
        vert,
        kind="quadratic",
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
    )
    proj_interpolator = interp1d(
        proj_dist,
        properties,
        kind="linear",
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
    )

    vert_xs = np.linspace(0, length, int(np.ceil(length / max_spacing)))
    proj_xs = (vert_xs[:-1] + vert_xs[1:]) / 2

    new_vert = vert_interpolator(vert_xs)
    new_proj = proj_interpolator(proj_xs)

    new_segments = np.array(list(_pairwise(new_vert)))
    return new_segments, new_proj


def _pairwise(iterable):
    """Convert an iterable, s, to (s0,s1), (s1,s2), (s2, s3), ..."""
    from itertools import tee

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
