import itertools
from typing import Tuple, List, Optional

import numpy as np

from trimesh.intersections import plane_lines
from trimesh.geometry import plane_transform
from trimesh.transformations import transform_points

from monty.json import MSONable
from scipy.spatial import Voronoi, ConvexHull

from pymatgen import Structure


class ReciprocalSlice(MSONable):

    def __init__(
            self,
            vertices: np.ndarray,
            transformation: np.ndarray,
            reciprocal_space: "ReciprocalSpace",
    ):
        self.vertices = vertices
        self.transformation = transformation
        self.reciprocal_space = reciprocal_space
        self._edges: Optional[List[Tuple[int, int]]] = None

    @property
    def edges(self):
        """
        Get the edges of the space as a List of tuples specifying the vertices.
        """
        if self._edges is None:
            hull = ConvexHull(self.vertices)
            self._edges = hull.simplices
        return self._edges

    @property
    def lines(self):
        """
        Get the lines defining the space as a list of two coordinates.
        """
        return self.vertices[np.array(self.edges)]


class ReciprocalSpace(MSONable):
    """
    Common representation of a reciprocal space.

    Attributes:
        reciprocal_lattice: The reciprocal lattice vectors.
    """

    def __init__(self, reciprocal_lattice: np.ndarray):
        """
        Args:
            reciprocal_lattice: The reciprocal lattice vectors.
        """
        self.reciprocal_lattice = reciprocal_lattice
        self.vertices: Optional[np.ndarray] = None
        self.faces: Optional[List[List[int]]] = None
        self._edges: Optional[List[int]] = None

    @classmethod
    def from_structure(cls, structure: Structure):
        """
        Initialise the class from a structure

        Args:
            structure: A structure.

        Returns:
            An instance of the class.
        """
        return cls(structure.lattice.reciprocal_lattice.matrix)

    @property
    def edges(self):
        """
        Get the edges of the space as a List of tuples specifying the vertices.
        """
        if self._edges is None:
            output = set()
            for face in self.faces:
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[i - 1]]))
                    output.add(edge)
            self._edges = list(set(output))
        return self._edges

    @property
    def lines(self):
        """
        Get the lines defining the space as a list of two coordinates.
        """
        return self.vertices[np.array(self.edges)]

    def get_reciprocal_slice(
        self,
        plane_normal: Tuple[int, int, int],
        distance: float = 0
    ) -> ReciprocalSlice:
        cart_normal = np.dot(plane_normal, self.reciprocal_lattice)
        cart_center = cart_normal * distance

        # get the intersections with the faces
        intersections, _ = plane_lines(
            cart_center, cart_normal, self.lines.transpose(1, 0, 2)
        )

        if len(intersections) == 0:
            raise ValueError("Plane does not intersect reciprocal cell")

        #  transform the intersections from 3D space to 2D coordinates
        transformation = plane_transform(origin=cart_center, normal=cart_normal)
        points = transform_points(intersections, transformation)[:, :2]

        return ReciprocalSlice(points, transformation, self)


class ReciprocalCell(ReciprocalSpace):
    """
    The reciprocal cell.

    Defined by the parallelepiped formed by the vector set (b1, b2, b3) and
    contains all the same information as the first Brillouin Zone.

    Attributes:
        faces: The coordinates for each face of the reciprocal cell.
    """

    def __init__(self, reciprocal_lattice: np.ndarray):
        """
        Args:
            reciprocal_lattice: The reciprocal lattice vectors.
        """
        super().__init__(reciprocal_lattice)
        vertices = [
            [0, 0, 0],  # 0
            [0, 0, 1],  # 1
            [0, 1, 0],  # 2
            [0, 1, 1],  # 3
            [1, 0, 0],  # 4
            [1, 0, 1],  # 5
            [1, 1, 0],  # 6
            [1, 1, 1]  # 7
        ]
        faces = [
            [0, 1, 3, 2],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 4, 6, 2],
            [1, 5, 7, 3]
        ]
        self.vertices = np.dot(np.array(vertices) - 0.5, reciprocal_lattice)
        self.faces = np.array(faces)


class WignerSeitzCell(ReciprocalSpace):
    """
    The first Brillioun Zone information.

    This is the Wigner–Seitz cell of the reciprocal lattice.
    """

    def __init__(self, reciprocal_lattice: np.ndarray):
        """
        Args:
            reciprocal_lattice: The reciprocal lattice vectors.
        """
        super().__init__(reciprocal_lattice)

        points = []
        for i, j, k in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
            points.append(np.dot([i, j, k], reciprocal_lattice))

        voronoi = Voronoi(points)

        #  find the bounded voronoi region vertices
        valid_vertices = set()
        for region in voronoi.regions:
            if -1 not in region:
                valid_vertices.update(region)

        # get the faces as the ridges that comprise the bounded region
        self.faces = [
            x for x in voronoi.ridge_vertices if set(x).issubset(valid_vertices)
        ]
        self.vertices = voronoi.vertices

        # get the center normals for all faces
        centers = []
        normals = []
        for face in self.faces:
            face_verts = self.vertices[face]
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

        self.centers = np.array(centers)
        self.normals = np.array(normals)