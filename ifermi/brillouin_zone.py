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
        """Summary
        
        Args:
            rlattvec (np.array): The lattice vector (b1, b2, b3) in reciprocal space.
        """
        self._rlattvec = rlattvec
        
        points = []
        for ijk0 in itertools.product(range(5), repeat=3):
            ijk = [i if i <= 2 else i - 5 for i in ijk0]
            points.append(rlattvec @ np.array(ijk))
        voronoi = sp.spatial.Voronoi(points)
        region_index = voronoi.point_region[0]
        vertex_indices = voronoi.regions[region_index]
        vertices = voronoi.vertices[vertex_indices, :]
        
        self._vertices = vertices
        
        # Compute a center and an outward-pointing normal for each of the facets
        # of the BZ
        facets = []
        for ridge in voronoi.ridge_vertices:
            if all(i in vertex_indices for i in ridge):
                facets.append(ridge)
        centers = []
        normals = []
        bz_corners = []
        
        for f in facets:
            corners = np.array([voronoi.vertices[i, :] for i in f])
            bz_corners.append(np.concatenate((corners, [corners[0]]), axis=0))
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

        # return box dimensions of the Brillouin zone for use in cropping later
        min_dimensions = [np.amin(vertices[:, 0]), np.amin(vertices[:, 1]), np.amin(vertices[:, 2])]
        max_dimensions = [np.amax(vertices[:, 0]), np.amax(vertices[:, 1]), np.amax(vertices[:, 2])]
        
        
        self._min_dimensions = min_dimensions
        self._max_dimensions = max_dimensions
        self._centers = centers
        self._normals = normals
        self._facets = facets
        self._bz_corners = bz_corners


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

        for i in [0,1]:
            corners = []
            for j in [0,1]:
                for k in [0,1]:
                    corners.append([k*np.linalg.norm([rlattvec[:,0]]),j*np.linalg.norm([rlattvec[:,1]]),i*np.linalg.norm([rlattvec[:,2]])])
            corners[-2], corners[-1] = corners[-1], corners[-2]
            corners.append(corners[0])
            faces.append(corners)

        for j in [0,1]:
            corners = []
            for i in [0,1]:
                for k in [0,1]:
                    corners.append([i*np.linalg.norm([rlattvec[:,0]]),j*np.linalg.norm([rlattvec[:,1]]),k*np.linalg.norm([rlattvec[:,2]])])
            corners[-2], corners[-1] = corners[-1], corners[-2]
            corners.append(corners[0])
            faces.append(corners)

        self._faces = faces
