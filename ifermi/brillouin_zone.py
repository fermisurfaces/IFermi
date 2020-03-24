"""
    This module contains objects used for defining the brillioun zones of a given crystal structure.

    """
import numpy as np
import scipy as sp
import scipy.linalg as la
import itertools

from pymatgen.electronic_structure.bandstructure import BandStructure
from skimage import measure


class BrillouinZone(object):
    """An object which holds information for the Brillioun Zone. This is the Wigner–Seitz cell
        of the reciprocal lattice.
        """

    def __init__(self, rlattvec: np.array):
        """
        Args:
            rlattvec (np.array): The lattice vector (b1, b2, b3) in reciprocal space.
        """
        self._rlattvec = rlattvec

        vec1 = rlattvec[0]
        vec2 = rlattvec[1]
        vec3 = rlattvec[2]

        points = []
        for i, j, k in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
            points.append(i * vec1 + j * vec2 + k * vec3)

        voronoi = sp.spatial.Voronoi(points)

        to_return = []
        centers = []
        normals = []
        for r in voronoi.ridge_dict:

            if r[0] == 13 or r[1] == 13:
                to_return.append(np.array([voronoi.vertices[i] for i in voronoi.ridge_dict[r]]))

        for i in to_return:
            corners = np.array(i)

            center = corners.mean(axis=0)
            v1 = corners[0, :]
            for i in range(1, corners.shape[0]):
                v2 = corners[i, :]
                prod = np.cross(v1 - center, v2 - center)
                if not np.allclose(prod, 0.):
                    break
            if np.dot(center, prod) < 0.:
                prod = -prod
            centers.append(center)
            normals.append(prod)

        to_return = np.array(to_return)

        # # return box dimensions of the Brillouin zone for use in cropping later
        # # Must check whether doing this cropping before using centers and normals to find points in the BZ is more
        # # efficient
        # min_dim_list = np.array([np.array([np.amin(i[:, 0]), np.amin(i[:, 1]), np.amin(i[:, 2])]) for i in to_return])
        # max_dim_list = np.array([np.array([np.amax(i[:, 0]), np.amax(i[:, 1]), np.amax(i[:, 2])]) for i in to_return])
        # min_dimensions = [np.amin(min_dim_list[:, 0]), np.amin(min_dim_list[:, 1]), np.amin(min_dim_list[:, 2])]
        # max_dimensions = [np.amax(max_dim_list[:, 0]), np.amax(max_dim_list[:, 1]), np.amax(max_dim_list[:, 2])]
        # self._min_dimensions = min_dimensions
        # self._max_dimensions = max_dimensions

        # These parameters are used in cropping the Fermi-surface to the Brillouin zone

        self._centers = centers
        self._normals = normals
        self._bz_corners = to_return


class RecipCell(object):
    """
    An object which holds information for the reciprocal cell. The reciprocal cell is the paralellopiped formed
    by the vector set (b1, b2, b3) and contains all the same information as the Brillouin Zone.
    """

    def __init__(self, rlattvec: np.array):
        """Summary

        Args:
            lattvec (np.array): The lattice vector (b1, b2, b3) in reciprocal space.
        """
        self._rlattvec = rlattvec

        faces = []

        for i in [0, 1]:
            corners = []
            for j in [0, 1]:
                for k in [0, 1]:
                    corners.append([k * np.linalg.norm([rlattvec[0]]), j * np.linalg.norm([rlattvec[1]]),
                                    i * np.linalg.norm([rlattvec[2]])])
            corners[-2], corners[-1] = corners[-1], corners[-2]
            corners.append(corners[0])
            faces.append(corners)

        for j in [0, 1]:
            corners = []
            for i in [0, 1]:
                for k in [0, 1]:
                    corners.append([i * np.linalg.norm([rlattvec[0]]), j * np.linalg.norm([rlattvec[1]]),
                                    k * np.linalg.norm([rlattvec[2]])])
            corners[-2], corners[-1] = corners[-1], corners[-2]
            corners.append(corners[0])
            faces.append(corners)

        self._faces = faces