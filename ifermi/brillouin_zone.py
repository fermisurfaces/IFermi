"""
This module contains objects used for defining reciprocal periodic boundaries.
"""

import itertools

import numpy as np
from monty.json import MSONable
from scipy.spatial import Voronoi

from pymatgen import Structure


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

    @classmethod
    def from_structure(cls, structure: Structure):
        return cls(structure.lattice.reciprocal_lattice.matrix)


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

        face_vertices = [
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]],
            [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]],

            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]],

            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]],
        ]
        face_vertices = np.array(face_vertices) - 0.5
        self.faces = np.dot(face_vertices, reciprocal_lattice)


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

        self.faces = []
        centers = []
        normals = []
        for ridge_points, ridge_vertices in voronoi.ridge_dict.items():
            if ridge_points[0] == 13 or ridge_points[1] == 13:
                corners = [voronoi.vertices[i] for i in ridge_vertices]
                self.faces.append(np.array(corners))

        for corners in self.faces:
            center = corners.mean(axis=0)
            v1 = corners[0, :]
            for v2 in corners[1:]:
                prod = np.cross(v1 - center, v2 - center)
                if not np.allclose(prod, 0.0):
                    break

            if np.dot(center, prod) < 0.0:
                prod = -prod

            centers.append(center)
            normals.append(prod)

        self.centers = np.array(centers)
        self.normals = np.array(normals)
