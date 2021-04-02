"""Tools to generate Isolines and Fermi slices."""
import warnings
from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Tuple, Union

import numpy as np
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin

from ifermi.analysis import equivalent_vertices, longest_simple_paths
from ifermi.brillouin_zone import ReciprocalSlice

__all__ = ["Isoline", "FermiSlice", "process_lines", "interpolate_segments"]


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

    def __post_init__(self):
        # ensure all inputs are numpy arrays

        self.segments = np.array(self.segments)

        if self.properties is not None:
            self.properties = np.array(self.properties)

    @property
    def has_properties(self) -> float:
        """Whether the isoline has properties."""
        return self.properties is not None

    def scalar_projection(self, axis: Tuple[int, int, int]) -> np.ndarray:
        """
        Get scalar projection of properties onto axis.

        Args:
            axis: A (3, ) int array of the axis to project onto.
        """
        if not self.has_properties:
            raise ValueError("Isoline does not have segment properties.")

        if self.properties.ndim != 2:
            raise ValueError("Isoline does not have vector properties.")

        return np.dot(self.properties, axis)

    @property
    def properties_norms(self) -> np.ndarray:
        """(m, ) norm of isoline properties."""
        if not self.has_properties:
            raise ValueError("Isoline does not have segment properties.")

        if self.properties.ndim != 2:
            raise ValueError("Isoline does not have vector properties.")

        return np.linalg.norm(self.properties, axis=1)

    @property
    def properties_ndim(self) -> int:
        """Dimensionality of face properties."""
        if not self.has_properties:
            raise ValueError("Isoline does not have segment properties.")

        return self.properties.ndim

    def sample_uniform(self, spacing: float) -> np.ndarray:
        """
        Sample line segments uniformly.

        See the docstring for ``ifermi.analysis.sample_line_uniform`` for more details.

        Args:
            spacing: The spacing in Å^-1.

        Returns:
            A (n, ) int array containing the indices of uniformly spaced segments.
        """
        from ifermi.analysis import sample_line_uniform

        return sample_line_uniform(self.segments, spacing)


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
        """Number of isolines in the Fermi slice."""
        return sum(self.n_lines_per_spin.values())

    @property
    def n_lines_per_band(self) -> Dict[Spin, Dict[int, int]]:
        """
        Get number of lines for each band index for each spin channel.

        Returned as a dict of ``{spin: {band_idx: count}}``.
        """
        from collections import Counter

        n_surfaces = {}
        for spin, isosurfaces in self.isolines.items():
            n_surfaces[spin] = dict(Counter([s.band_idx for s in isosurfaces]))

        return n_surfaces

    @property
    def n_lines_per_spin(self) -> Dict[Spin, int]:
        """
        Get number of lines per spin channel.

        Returned as a dict of ``{spin: count}``.
        """
        return {spin: len(surfaces) for spin, surfaces in self.isolines.items()}

    @property
    def has_properties(self) -> bool:
        """Whether all isolines have segment properties."""
        return all([all([i.has_properties for i in s]) for s in self.isolines.values()])

    @property
    def spins(self) -> Tuple[Spin]:
        """The spin channels in the Fermi slice."""
        return tuple(self.isolines.keys())

    @property
    def properties_ndim(self) -> int:
        """Dimensionality of isoline properties."""
        if not self.has_properties:
            raise ValueError("Isolines don't have properties.")

        ndims = [i.properties_ndim for v in self.isolines.values() for i in v]

        if len(set(ndims)) != 1:
            warnings.warn(
                "Ioslines have different property dimensions, using the largest."
            )

        return max(ndims)

    def all_lines(
        self, spins: Optional[Union[Spin, Collection[Spin]]] = None
    ) -> List[np.ndarray]:
        """
        Get the segments for all isolines.

        Args:
            spins: One or more spin channels to select. Default is all spins available.

        Returns:
            A list of segments arrays.
        """
        if not spins:
            spins = self.spins
        elif isinstance(spins, Spin):
            spins = [spins]

        return [line.segments for spin in spins for line in self.isolines[spin]]

    def all_properties(
        self,
        spins: Optional[Union[Spin, Collection[Spin]]] = None,
        projection_axis: Optional[Tuple[int, int, int]] = None,
        norm: bool = False,
    ) -> List[np.ndarray]:
        """
        Get the properties for all isolines.

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
            for isosurface in self.isolines[spin]:
                if projection_axis is not None:
                    projections.append(isosurface.scalar_projection(projection_axis))
                elif norm:
                    projections.append(isosurface.properties_norms)
                else:
                    projections.append(isosurface.properties)

        return projections

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
        from collections import defaultdict

        from trimesh import Trimesh
        from trimesh.intersections import mesh_multiplane

        cart_normal = np.dot(
            plane_normal, fermi_surface.reciprocal_space.reciprocal_lattice
        )
        cart_origin = cart_normal * distance

        isolines = defaultdict(list)
        for spin in fermi_surface.spins:

            for isosurface in fermi_surface.isosurfaces[spin]:
                mesh = Trimesh(vertices=isosurface.vertices, faces=isosurface.faces)
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
                    path_properties = None
                    if isosurface.has_properties:
                        path_properties = isosurface.properties[path_faces]
                        path_segments, path_properties = interpolate_segments(
                            path_segments, path_properties, 0.001
                        )
                    isoline = Isoline(
                        segments=path_segments,
                        band_idx=isosurface.band_idx,
                        properties=path_properties,
                    )
                    isolines[spin].append(isoline)

        reciprocal_slice = fermi_surface.reciprocal_space.get_reciprocal_slice(
            plane_normal, distance
        )

        return FermiSlice(dict(isolines), reciprocal_slice, fermi_surface.structure)

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
        properties = np.concatenate([[properties[0]], properties, [properties[-1]]])

    vert_interpolator = interp1d(
        vert_dist,
        vert,
        kind="cubic",
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
