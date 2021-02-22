"""Isosurface and isoline analysis functions."""

import warnings
from typing import List, Tuple, Union

import numpy as np

from ifermi.defaults import KTOL

__all__ = [
    "isosurface_area",
    "isosurface_dimensionality",
    "connected_subsurfaces",
    "average_properties",
    "line_orientation",
    "plane_orientation",
    "connected_images",
    "equivalent_surfaces",
    "equivalent_vertices",
    "sample_line_uniform",
    "sample_surface_uniform",
    "longest_simple_paths",
]


def isosurface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Calculate the area of an isosurface.

    Args:
        vertices: A (n, 3) float array of the vertices in the isosurface.
        faces: A (m, 3) int array of the faces of the isosurface.

    Returns:
        The area of the isosurface, in Å^-2.
    """
    from trimesh import Trimesh

    mesh = Trimesh(vertices=vertices, faces=faces)
    return mesh.area


def average_properties(
    vertices: np.ndarray, faces: np.ndarray, properties: np.ndarray, norm: bool = False
) -> Union[float, np.ndarray]:
    """
    Average property across an isosurface.

    Args:
        vertices: A (n, 3) float array of the vertices in the isosurface.
        faces: A (m, 3) int array of the faces of the isosurface.
        properties: A (m, ...) array of the face properties as scalars or vectors.
        norm: Whether to average the norm of the properties (vector properties only).

    Returns:
        The averaged property.
    """
    from trimesh import Trimesh

    mesh = Trimesh(vertices=vertices, faces=faces)

    if norm and properties.ndim > 1:
        properties = np.linalg.norm(properties, axis=1)

    face_areas = mesh.area_faces

    # face_areas has to have same number of dimensions as property
    face_areas = face_areas.reshape(face_areas.shape + (1,) * (properties.ndim - 1))

    return np.sum(properties * face_areas, axis=0) / mesh.area


def connected_subsurfaces(
    vertices: np.ndarray, faces: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Find connected sub-surfaces (those that share edges).

    Args:
        vertices: A (n, 3) float array of the vertices in the isosurface.
        faces: A (m, 3) int array of the faces of the isosurface.

    Returns:
        A list of of (vertices, faces) for each sub-surface.
    """
    from trimesh import Trimesh

    mesh = Trimesh(vertices=vertices, faces=faces)
    connected_meshes = mesh.split(only_watertight=False)
    return [(m.vertices, m.faces) for m in connected_meshes]


def equivalent_surfaces(surfaces_vertices: List[np.ndarray], tol=KTOL) -> np.ndarray:
    """

    Note: This function expects the vertices of each surface to only belong to a single
    connected mesh, and the vertices should cover a 3x3x3 supercell, where the
    coordinates from -0.5 to 0.5 indicate the center image.

    Args:
        surfaces_vertices: A (m, ) list of (n, 3) float arrays containing the vertices
            for each surface, in fractional coordinates.
        tol: A tolerance for evaluating whether two vertices are equivalent.

    Returns:
        A (m, ) int array that maps each each surface in the original surface array to
        its equivalent.
    """
    from ifermi.kpoints import kpoints_to_first_bz

    round_dp = int(np.log10(1 / tol))

    # loop through all surfaces and find those that are present in the center cell
    mapping = {}
    for surface_idx, vertices in enumerate(surfaces_vertices):
        # in_center_cell = np.all((vertices >= -0.5) & (vertices < 0.5), axis=1)
        center_cell = np.all(vertices >= -0.5, axis=1) & np.all(vertices < 0.5, axis=1)

        if np.any(center_cell):
            vertices = vertices[center_cell].round(round_dp)
            mapping[surface_idx] = set(list(map(tuple, vertices)))

    mapping_order = []
    # now find which other surfaces map to the those that go through the center cell
    for surface_idx, vertices in enumerate(surfaces_vertices):
        if surface_idx in mapping:
            mapping_order.append(surface_idx)
            continue

        vertices = kpoints_to_first_bz(vertices).round(round_dp)
        vertices_set = set(map(tuple, vertices))

        match = None
        for mapping_idx, mapping_vertices in mapping.items():
            if len(mapping_vertices.intersection(vertices_set)) > 0:
                match = mapping_idx
                continue

        if not match:
            warnings.warn("Could not map surface")
            match = surface_idx
        mapping_order.append(match)

    return np.array(mapping_order)


def isosurface_dimensionality(
    fractional_vertices: np.ndarray, faces: np.ndarray
) -> Tuple[str, Tuple[int, int, int]]:
    """
    Calculate isosurface properties a single isosurface (fully connected).

    The vertices must cover a 3x3x3 supercell and must not have been trimmed to fit
    inside the reciprocal lattice.

    Note: This function expects the vertices to only belong to a single connected
    mesh, and the vertices should cover a 3x3x3 supercell, where the coordinates
    from -0.5 to 0.5 indicate the center image.

    Args:
        fractional_vertices: A (n, 3) float array of the vertices in the isosurface
            in fractional coordinates.
        faces: A (m, 3) int array of the faces of the isosurface.

    Returns:
        The dimensionality and (n, 3) int array orientation of the isosurface.
    """
    from trimesh import Trimesh

    if len(connected_subsurfaces(fractional_vertices, faces)) != 1:
        raise ValueError("isosurface contains multiple subsurfaces")

    images = connected_images(fractional_vertices)

    rank = np.linalg.matrix_rank(images)
    orientation = None

    if rank == 1:
        orientation = line_orientation(images)
        dimensionality = "2D"
    elif rank == 2:
        orientation = plane_orientation(images)

        # use euler number to decide if mesh is a plane or multiple tubes
        euler_number = Trimesh(vertices=fractional_vertices, faces=faces).euler_number
        if euler_number == 1:
            dimensionality = "1D"
        else:
            dimensionality = "quasi-2D"
    elif rank == 0:
        dimensionality = "3D"
    else:
        dimensionality = "quasi-3D"

    return dimensionality, orientation


def connected_images(
    fractional_vertices: np.ndarray, tol=KTOL
) -> List[Tuple[int, int, int]]:
    """
    Find the images a set of vertices is connected to.

    Note: This function expects the vertices to only belong to a single connected
    mesh, and the vertices should cover a 3x3x3 supercell, where the coordinates
    from -0.5 to 0.5 indicate the center image.

    Args:
        fractional_vertices: A (n, 3) float array of the vertices in the isosurface
            in fractional coordinates.
        tol: A tolerance for evaluating whether two vertices are equivalent.

    Returns:
        A (n, 3) int array of the images across which the mesh is connected
        periodically.
    """
    from collections import defaultdict

    from ifermi.kpoints import kpoints_to_first_bz

    # shift vertices so that the vertices in the center image fall between 0-1
    vertices_first_bz = kpoints_to_first_bz(fractional_vertices) + 0.5
    vertices = fractional_vertices + 0.5

    # we will be filtering the vertices based on whether they fall into a particular
    # periodic image. To avoid having to recalculate whether certain coordinates
    # fall within certain images we will cache the filters
    coordinate_filters = defaultdict(dict)

    def filter_coordinates(axis: int, image: int) -> np.ndarray:
        """Filter vertices on whether the coordinate fall within an image."""
        if axis not in coordinate_filters or image not in coordinate_filters[axis]:
            coordinate_filters[axis][image] = (vertices[:, axis] > image) & (
                vertices[:, axis] <= image + 1
            )
        return coordinate_filters[axis][image]

    def mesh_in_image(image, vertex=None, return_vertices=False):
        """
        Check if any vertices are in a periodic image.

        If a vertex is provided, the vertices that fall within the periodic image
        will be mapped back to the center image and used to determine if the provided
        vertex is among them. If it isn't, this indicates that the connection is
        to a different mesh than the one in the center image.
        """
        mask = (
            filter_coordinates(0, image[0])
            & filter_coordinates(1, image[1])
            & filter_coordinates(2, image[2])
        )
        if vertex is None and not return_vertices:
            return np.any(mask)
        elif vertex is None:
            return np.any(mask), vertices[mask]

        vertices_in_image = vertices_first_bz[mask]
        is_in_image = np.any(np.linalg.norm(vertices_in_image - vertex, axis=1) < tol)
        if return_vertices:
            return is_in_image, vertices_in_image
        else:
            return is_in_image

    if not mesh_in_image((0, 0, 0)):
        # if none of the vertices are in the center cell then skip this mesh
        return []

    # choose a vertex of the mesh that is in the center cell and that will be used
    # to identify this particular mesh
    mesh_vertex = mesh_in_image((0, 0, 0), return_vertices=True)[1][0]

    found_connections = []
    for i, j, k in np.ndindex((3, 3, 3)):
        to_image = (i - 1, j - 1, k - 1)

        if mesh_in_image(to_image, vertex=mesh_vertex):
            found_connections.append(to_image)

    return found_connections


def line_orientation(images: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """
    Get the orientation (direction vector) from a list of rank 1 connected images.

    Args:
        images: (n, 3) int array of the images.

    Returns:
        The orientation vector as a (3, ) int array.
    """
    from pymatgen.core.lattice import get_integer_index

    vertices = np.array(images)
    g = vertices.sum(axis=0) / vertices.shape[0]
    _, _, vh = np.linalg.svd(vertices - g)  # run singular value decomposition

    return get_integer_index(vh[0, :])  # return line of best fit


def plane_orientation(images: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """
    Get the orientation (surface normal) from a list of rank 2 connected images.

    Args:
        images: (n, 3) int array of the images.

    Returns:
        The surface normal as a (3, ) int array.
    """
    from pymatgen.core.lattice import get_integer_index

    vertices = np.array(images)
    g = vertices.sum(axis=0) / vertices.shape[0]
    _, _, vh = np.linalg.svd(vertices - g)  # run singular value decomposition
    return get_integer_index(vh[2, :])  # return unitary norm


def sample_surface_uniform(
    vertices: np.ndarray, faces: np.ndarray, grid_size: float
) -> np.ndarray:
    """
    Sample isosurface faces uniformly.

    The algorithm works by:

    1. Splitting the mesh into a uniform grid with block sizes determined by
       ``grid_size``.
    2. For each cell in the grid, finds whether the center of any faces falls within the
       cell.
    3. If multiple face centers fall within the cell, it picks the closest one to the
       center of the cell. If no face centers fall within the cell then the cell is
       ignored.
    4. Returns the indices of all the faces that have been selected.

    This algorithm is not well optimised for small grid sizes.

    Args:
        vertices: A (n, 3) float array of the vertices in the isosurface.
        faces: A (m, 3) int array of the faces of the isosurface.
        grid_size: The grid size in Å^-1.

    Returns:
        A (k, ) int array containing the indices of uniformly spaced faces.
    """
    face_verts = vertices[faces]
    centers = face_verts.mean(axis=1)
    min_coords = np.min(centers, axis=0)
    max_coords = np.max(centers, axis=0)

    lengths = max_coords - min_coords
    min_coords -= lengths * 0.2
    max_coords += lengths * 0.2

    n_grid = np.ceil((max_coords - min_coords) / grid_size).astype(int)

    center_idxs = np.arange(len(centers))

    selected_faces = []
    for cell_image in np.ndindex(tuple(n_grid)):
        cell_image = np.array(cell_image)
        cell_min = min_coords + cell_image * grid_size
        cell_max = min_coords + (cell_image + 1) * grid_size

        # find centers that fall within the cell
        within = np.all(centers > cell_min, axis=1) & np.all(centers < cell_max, axis=1)

        if not np.any(within):
            continue

        # get the indexes of those centers
        within_idx = center_idxs[within]

        # of these, find the center that is closest to the center of the cell
        distances = np.linalg.norm((cell_max + cell_min) / 2 - centers[within], axis=1)
        select_idx = within_idx[np.argmin(distances)]

        selected_faces.append(select_idx)

    return np.array(selected_faces)


def sample_line_uniform(segments: np.ndarray, spacing: float) -> np.ndarray:
    """
    Sample line segments to a consistent density.

    Note: the segments must be ordered so that they are adjacent.

    Args:
        segments: (n, 2, 2) float array of line segments.
        spacing: The desired spacing in Å^-1.

    Returns:
        A (m, ) int array containing the indices of uniformly spaced segments.
    """
    segment_lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)

    # total line length
    line_length = np.sum(segment_lengths)

    # this is the distance from the center of each segment to the beginning of the line
    center_lengths = np.cumsum(segment_lengths) - segment_lengths / 2

    # find the distance of each arrow from the beginning of the line if ideally spaced
    narrows = max(1, int(np.floor(line_length / spacing)))

    # recalculate spacing so that all arrows are equally spaced
    spacing = line_length / narrows

    # shift = (line_length - (narrows * spacing)) / 2
    ideal_pos = np.linspace(0, (narrows - 1) * spacing, narrows) + spacing / 2

    return np.argmin(np.abs(ideal_pos[:, None] - center_lengths[None, :]), axis=1)


def equivalent_vertices(vertices: np.ndarray, tol: float = KTOL) -> np.ndarray:
    """
    Find vertices that are equivalent (closer than a tolerance).

    Note that the algorithm used is effectively recursive. If vertex a is within the
    tolerance of b, and b is within the tolerance of c, even if a and c and not within
    the tolerance, a, b, and c will be considered equivalent.

    Args:
        vertices: (n, 2) or (n, 3) float array of the vertices.
        tol: The distance tolerance for equivalence.

    Returns:
        (n, ) int array that maps each each vertex in the original vertex array to its
        equivalent.
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


def longest_simple_paths(vertices: np.ndarray, edges: np.ndarray) -> List[np.ndarray]:
    """
    Find the shortest paths that go through all vertices.

    The lines are broken up into the connected sublines. Note this function is only
    designed to work with connected sublines that are either simple cycles or
    chains with no branches.

    Args:
        vertices: (n, 2) float array of the line vertices.
        edges: (m, 2) int array of the edges.

    Returns:
        A list of (k, ) int arrays (one for each connected subline) specifying the path.
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
                raise ValueError("Path is not unique; unable to find path")

            start, end = ends
            simple_paths = list(nx.all_simple_paths(subgraph, start, end))
            longest_path = sorted(simple_paths, key=lambda x: len(x))[-1]

        if set(longest_path) != set(subgraph.nodes):
            raise ValueError("Path does not cover all vertices.")

        paths.append(longest_path)

    return paths
