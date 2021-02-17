"""Functions to analyze Fermi surface and Fermi slice data."""
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

__all__ = [
    "isosurface_area",
    "isosurface_properties",
    "average_projections",
    "line_orientation",
    "plane_orientation",
    "connected_images",
    "SurfaceProperties"
]


def isosurface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Calculate the area of an iso-surface.

    Args:
        vertices: The vertices in the iso-surface as a numpy array with the shape
            (nvertices, 3).
        faces: The faces of the iso-surface as an integer numpy array with the shape
            (nfaces, 3).

    Returns:
        The area of the iso-surface, in Å^-2.
    """
    from trimesh import Trimesh

    mesh = Trimesh(vertices=vertices, faces=faces)
    return mesh.area


def average_projections(
    vertices: np.ndarray, faces: np.ndarray, projections: np.ndarray, norm: bool = False
) -> Union[float, np.ndarray]:
    """
    Average projections across an iso-surface.

    Args:
        vertices: The vertices in the iso-surface as a numpy array with the shape
            (nvertices, 3).
        faces: The faces of the iso-surface as an integer numpy array with the shape
            (nfaces, 3).
        projections: The projections for each face, given as a numpy array of
            with the shape (nfaces, ...). The projections can be scalar or vectors.
        norm: Whether to average the norm of the projections. Only applicable for
            vector projections.

    Returns:
        The averaged projection value. The returned value will have the same number
        of dimensions as the original projections (unless ``norm`` is set to True.
    """
    from trimesh import Trimesh

    mesh = Trimesh(vertices=vertices, faces=faces)

    if norm:
        projections = np.linalg.norm(projections, axis=1)

    return np.sum(projections * mesh.area_faces, axis=0) / mesh.area


@dataclass
class SurfaceProperties:
    """
    Class containing information about surface properties.

    Note: All properties are given with respect to the parallelepiped reciprocal
    lattice.

    Args:
        area: The are of the surface.
        dimensionality: The dimensionality of the Fermi surface.
        orientation: The orientation (surface normal or vector) if the dimensionality is
            one or two dimensional.
    """

    area: float
    dimensionality: str
    orientation: Tuple[int, int, int]


def isosurface_properties(
    vertices: np.ndarray,
    faces: np.ndarray,
    reciprocal_lattice: np.ndarray,
) -> List[SurfaceProperties]:
    """
    Calculate iso-surface properties for connected sub-surfaces.

    The vertices must cover a 3x3x3 supercell and must not have been trimmed to fit
    inside the reciprocal lattice.

    Args:
        vertices: The vertices of the mesh.
        faces: The faces of the mesh.
        reciprocal_lattice: The reciprocal lattice matrix.

    Returns:
        A list of SurfaceProperties for each connected sub-surface.
    """
    from trimesh import Trimesh

    # convert vertices to fractional coordinates and shift them so that
    # the vertices in the center image fall between 0-1
    vertices = np.dot(vertices, np.linalg.inv(reciprocal_lattice)) + 0.5
    mesh = Trimesh(vertices=vertices, faces=faces)
    connected_meshes = mesh.split(only_watertight=False)

    mesh_data = []
    for connected_mesh in connected_meshes:
        images = connected_images(connected_mesh.vertices, connected_mesh.faces)
        rank = np.linalg.matrix_rank(images)

        orientation = None
        if rank == 1:
            orientation = line_orientation(images)
            dimensionality = "2D"
        elif rank == 2:
            orientation = plane_orientation(images)
            dimensionality = "1D"
        elif rank == 0:
            dimensionality = "3D"
        else:
            dimensionality = "quasi-3D"

        area = connected_mesh.area
        properties = SurfaceProperties(
            area=area, dimensionality=dimensionality, orientation=orientation
        )
        mesh_data.append(properties)

    return mesh_data


def connected_images(
    fractional_vertices: np.ndarray, faces: np.ndarray
) -> List[Tuple[int, int, int]]:
    """
    Find the images a set of vertices is connected to.

    Note: This function expects the vertices to only belong to a single connected
    mesh, and the vertices should cover a 3x3x3 supercell, where the coordinates
    from 0 to 1 indicate the center image.

    Args:
        fractional_vertices: The vertices in fractional coordinates.
        faces: The faces of the mesh.

    Returns:
        The connectivity of the mesh, as a list of periodic images that represent
        periodic boundary conditions across which the mesh is connected.
    """
    from collections import defaultdict

    from ifermi.kpoints import kpoints_to_first_bz

    vertices = fractional_vertices[np.unique(faces)]
    vertices_first_bz = kpoints_to_first_bz(vertices - 0.5) + 0.5

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

        vertices_in_image = vertices_first_bz[mask]
        is_in_image = np.any(np.allclose(vertices_in_image, vertex))
        if return_vertices:
            return is_in_image, vertices_in_image
        else:
            return is_in_image

    if not mesh_in_image((0, 0, 0), vertices):
        # if none of the vertices are in the center cell then skip this mesh
        return []

    # choose a vertex of the mesh that is in the center cell and that will be used
    # to identify this particular mesh
    _, mesh_vertex = mesh_in_image((0, 0, 0), vertices, return_vertices=True)

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
        images: The images.

    Returns:
        The orientation vector.
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
        images: The images.

    Returns:
        The surface normal.
    """
    from pymatgen.core.lattice import get_integer_index

    vertices = np.array(images)
    g = vertices.sum(axis=0) / vertices.shape[0]
    _, _, vh = np.linalg.svd(vertices - g)  # run singular value decomposition
    return get_integer_index(vh[2, :])  # return unitary norm
