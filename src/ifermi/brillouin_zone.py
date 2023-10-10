"""Brillouin zone and slice geometries."""

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from monty.json import MSONable
from pymatgen.core.structure import Structure

__all__ = ["ReciprocalSlice", "ReciprocalCell", "WignerSeitzCell"]


@dataclass
class ReciprocalSlice(MSONable):
    """
    A slice along a pane in reciprocal space.

    Attributes:
        reciprocal_space: The reciprocal space that the slice belongs to.
        vertices: A (n, 2) float array of the vertices defining the intersection of the
            slice with the Brillouin zone.
        transformation: (n, 4, 4) float array of the transformation that maps points in
            the 3D Brillouin zone to points on the 2D slice.
    """

    reciprocal_space: "ReciprocalCell"
    vertices: np.ndarray
    transformation: np.ndarray
    _edges: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        # ensure all inputs are numpy arrays

        self.vertices = np.array(self.vertices)
        self.transformation = np.array(self.transformation)

    @property
    def edges(self) -> np.ndarray:
        """A (n, 2) int array defining the cell edges."""
        from scipy.spatial import ConvexHull

        if self._edges is None:
            hull = ConvexHull(self.vertices)
            self._edges = hull.simplices
        return self._edges

    @property
    def lines(self) -> np.ndarray:
        """A (n, 2, 2) float array defining the cell boundaries."""
        return self.vertices[self.edges]


@dataclass
class ReciprocalCell(MSONable):
    """
    A parallelepiped reciprocal lattice cell.

    Attributes:
        reciprocal_lattice: A (3, 3) float array of the reciprocal lattice vectors.
        vertices: A (n, 3) float array of the vertices of the Brillouin zone edges.
        faces: A (m, 3) int array of the faces of the Brillouin zone.
        centers: A (m, 3) float array of the face centers.
        normals: A (m, 3) float array of the face normal.
    """

    reciprocal_lattice: np.ndarray
    vertices: np.ndarray
    faces: List[List[int]]
    centers: np.ndarray
    normals: np.ndarray
    _edges: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        # ensure all inputs are numpy arrays

        self.reciprocal_lattice = np.array(self.reciprocal_lattice)
        self.vertices = np.array(self.vertices)
        self.centers = np.array(self.centers)
        self.normals = np.array(self.normals)

    @classmethod
    def from_structure(cls, structure: Structure) -> "ReciprocalCell":
        """
        Initialise the reciprocal cell from a structure.

        Args:
            structure: A structure.

        Returns:
            The reciprocal cell.
        """
        reciprocal_lattice = structure.lattice.reciprocal_lattice.matrix
        vertices = [
            [0, 0, 0],  # 0
            [0, 0, 1],  # 1
            [0, 1, 0],  # 2
            [0, 1, 1],  # 3
            [1, 0, 0],  # 4
            [1, 0, 1],  # 5
            [1, 1, 0],  # 6
            [1, 1, 1],  # 7
        ]
        faces = [
            [0, 1, 3, 2],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 4, 6, 2],
            [1, 5, 7, 3],
        ]
        vertices = np.dot(np.array(vertices) - 0.5, reciprocal_lattice)

        # get the center normals for all faces
        centers = []
        normals = []
        for face in faces:
            face_verts = vertices[face]
            center = face_verts.mean(axis=0)

            v1 = face_verts[0] - center
            for v2 in face_verts[1:]:
                normal = np.cross(v1, v2 - center)
                if not np.allclose(normal, 0.0):
                    break

            if np.dot(center, normal) < 0.0:
                normal = -normal

            centers.append(center)
            normals.append(normal)

        centers = np.array(centers)
        normals = np.array(normals)
        return cls(reciprocal_lattice, vertices, faces, centers, normals)

    @property
    def edges(self) -> np.ndarray:
        """A (n, 2) int array defining the cell edges."""
        if self._edges is None:
            output = set()
            for face in self.faces:
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[i - 1]]))
                    output.add(edge)
            self._edges = np.array(list(set(output)))
        return self._edges

    @property
    def lines(self) -> np.ndarray:
        """A (n, 2, 3) float array defining the cell boundaries."""
        return self.vertices[self.edges]

    def get_reciprocal_slice(
        self, plane_normal: Tuple[int, int, int], distance: float = 0
    ) -> ReciprocalSlice:
        """
        Get a reciprocal slice through the Brillouin zone.

        Reciprocal slice defined by the intersection of a plane with the lattice.

        Args:
            plane_normal: (3, ) int array of the plane normal in fractional indices.
            distance: The distance from the center of the Brillouin zone (Γ-point).

        Returns:
            The reciprocal slice.
        """
        from trimesh import transform_points
        from trimesh.geometry import plane_transform
        from trimesh.intersections import plane_lines

        cart_normal = np.dot(plane_normal, self.reciprocal_lattice)
        cart_center = cart_normal * distance

        # get the intersections with the faces
        intersections, _ = plane_lines(
            cart_center, cart_normal, self.lines.transpose(1, 0, 2)
        )

        if len(intersections) == 0:
            raise ValueError("Plane does not intersect reciprocal cell")

        # transform the intersections from 3D space to 2D coordinates
        transformation = plane_transform(origin=cart_center, normal=cart_normal)
        points = transform_points(intersections, transformation)[:, :2]

        return ReciprocalSlice(self, points, transformation)


@dataclass
class WignerSeitzCell(ReciprocalCell):
    """
    WignerSeitz cell of the reciprocal lattice.

    Attributes:
        reciprocal_lattice: A (3, 3) float array of the reciprocal lattice vectors.
        vertices: A (n, 3) float array of the vertices of the Brillouin zone edges.
        faces: A (m, 3) int array of the faces of the Brillouin zone.
        centers: A (m, 3) float array of the face centers.
        normals: A (m, 3) float array of the face normal.
    """

    @classmethod
    def from_structure(cls, structure: Structure) -> "WignerSeitzCell":
        """
        Initialise the Wigner–Seitz cell from a structure.

        Args:
            structure: A structure.

        Returns:
            An instance of the cell.
        """
        from scipy.spatial.qhull import Voronoi

        reciprocal_lattice = structure.lattice.reciprocal_lattice.matrix

        points = []
        for i, j, k in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
            points.append(np.dot([i, j, k], reciprocal_lattice))

        voronoi = Voronoi(points)

        # find the bounded voronoi region vertices
        valid_vertices = set()
        for region in voronoi.regions:
            if -1 not in region:
                valid_vertices.update(region)

        # get the faces as the ridges that comprise the bounded region
        faces = [x for x in voronoi.ridge_vertices if set(x).issubset(valid_vertices)]
        vertices = voronoi.vertices

        # get the center normals for all faces
        centers = []
        normals = []
        for face in faces:
            face_verts = vertices[face]
            center = face_verts.mean(axis=0)

            v1 = face_verts[0] - center
            for v2 in face_verts[1:]:
                normal = np.cross(v1, v2 - center)
                if not np.allclose(normal, 0.0):
                    break

            if np.dot(center, normal) < 0.0:
                normal = -normal

            centers.append(center)
            normals.append(normal)

        centers = np.array(centers)
        normals = np.array(normals)
        return cls(reciprocal_lattice, vertices, faces, centers, normals)
