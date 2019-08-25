# Label by negative or positive, slice, place into new grid
import numpy as np
import scipy as sp
import scipy.linalg as la
import itertools

from pymatgen.electronic_structure.bandstructure import BandStructure
from skimage import measure

class FermiSurface(object):

    """An object which holds information relevent to the Fermi Surface. It only stores information at k-space points
    where energy(K) == fermi energy
    """
    
    def __init__(self, bs: BandStructure, hdims: list, rlattvec,
                 mu:float = 0, soc: bool = False, doping = None) -> None:
        """
        Args:
            bs (BandStructure): Description
            mu (float, optional): Enegy offset from the Fermi energy at which 
                the iso-surface is defined. Useful for visualising the effect of
                dopants on the shape of the surface. 
            hdims (list): Description
            soc (bool, optional): Description
            doping (None, optional): Description
            k_dim (int): Description
            fermi_level (int): The Fermi energy
            iso_surface (np.array): A List containing all parameters for each surface at the Fermi surface.
                                    The index i will access [verts, faces, normals, values] for surface i.  
            n_bands (int): Number of bands which cross the Fermi-Surface
            is_spin_polarised (bool, optional): set to True if spin polarised.
        """

        if doping is not None:
            # perform some calculation for mu from doping (how is this done??)
            mu = 0.1
            
        self._fermi_level = bs.efermi + mu

        self._kpoints = np.array([k.frac_coords for k in bs.kpoints])
        
        self._hdims = hdims

        dims = 2 * hdims + 1

        self._dims = dims

        self._k_dim = (dims[0], dims[1], dims[2])

        self._rlattvec = rlattvec

        # self.expand_bands(bs)

        # self.trim_energies(bs)

        # self.rearrange_bands(bs)

        self._spacing = (np.linalg.norm(bs.lattice_rec._matrix[:,0]) / dims[0], np.linalg.norm(bs.lattice_rec._matrix[:,1]) / dims[1], np.linalg.norm(bs.lattice_rec._matrix[:,2]) / dims[2])

        self.compute_isosurfaces(bs)       

        self._n_bands = len(self._iso_surface)

        self._soc = soc

        self._doping = doping

        self._structure = bs.structure



    @property
    def k_dim(self):
        return self._k_dim

    @property
    def fermi_level(self):
        return self._fermi_level

    @property
    def iso_surface(self):
        return self._iso_surface

    @property
    def n_bands(self):
        return self._n_bands

    @property
    def is_spin_polarised(self):
        return self._is_spin_polarised

    def compute_isosurfaces(self, bs):
        total_data = []
        iso_surface = []

        for spin in bs.bands.keys():

            ebands = bs.bands[spin]

            ebands -= self._fermi_level

            for  band in ebands:

                emax = np.nanmax(band)
                emin = np.nanmin(band)

                if emax > 0 and emin < 0:
                    total_data.append(band.reshape(np.array(self._dims)))

        for band_index, band_data in enumerate(total_data):

            verts, faces, normals, values = measure.marching_cubes_lewiner(band_data, 0,
                                                                           self._spacing)

            iso_surface.append([verts, faces, normals, values])


        self._iso_surface = iso_surface

    def expand_bands(self, bs):
        for spin in bs.bands.keys():
            ebands = bs.bands[spin]

            super_ebands = []

            dims = self._dims
            hdims = self._hdims

            new_ebands = []

            A = (-1, 0, 1)

            super_kpoints = np.array([], dtype=np.int64).reshape(0,3)

            for i, j, k in itertools.product(A, A, A):

                super_kpoints = np.concatenate((super_kpoints, np.array(self._kpoints) + [i,j,k]), axis = 0)

            sort_idx = np.lexsort((super_kpoints[:, 2], super_kpoints[:, 1], super_kpoints[:, 0]))

            final_kpoints = super_kpoints[sort_idx]

            for band in ebands:
                A = (-1, 0, 1)

                super_band = np.array([], dtype=np.int64)  

                for i, j, k in itertools.product(A, A, A):

                    super_band = np.concatenate((super_band, np.array(band)), axis=0)

                super_ebands.append(super_band[sort_idx])

            bs.bands[spin] = np.array(super_ebands)

        self._kpoints = final_kpoints

        self._dims = 3*dims

        self._hdims = 3*hdims


    def rearrange_bands(self, bs):

        sort_idx = np.lexsort((self._kpoints[:, 2], self._kpoints[:, 1], self._kpoints[:, 0]))

        for spin in bs.bands.keys():
            ebands = bs.bands[spin]

            new_ebands = []
           
            for band in ebands:

                band_sorted = band[sort_idx]
                new_ebands.append(band_sorted)

            bs.bands[spin] = np.array(new_ebands)

    
    def trim_energies(self, bs):

        bz = BrillouinZone(self._rlattvec)

        for spin in bs.bands.keys():
            ebands = bs.bands[spin]

            #Remove all points outside the BZ
            for i, ijk in enumerate(self._kpoints):
                abc = np.array(ijk)
                xyz = self._rlattvec @ abc
                for c, n in zip(bz._centers, bz._normals):
                    if np.dot(xyz - c, n) > 0.:
                        ebands[:, i] = None
                        break

            bs.bands[spin]=ebands



    def project_data(self, proj_plane: tuple):

        projected_band = []

        for i, band in enumerate(self._iso_surface):
            verts = band[0]
            faces = band[1] 

            projected_verts = []

            for vertex in verts:
                projected_verts.append(project(vertex, proj_plane))

            projected_band.append([projected_verts, faces])

        return projected_band

    def slice_data(self, bs, slice_plane: tuple, contour):

        for spin in bs.bands.keys():

            ebands = bs.bands[spin]

            plane_bands = []

            dis_array = [plane_dist(np.append(proj_plane, -contour), i) for i in self._kpoints]
        
            for i, j in enumerate(slice_array):
                if not j == 0:
                    if i==0:
                        sort_indx = np.lexsort(dis_array[dis_array[:,0].argsort()])
                        plane_mesh = [1, mesh[1], mesh[2]]
                    if i==1:
                        sort_indx = np.lexsort(dis_array[dis_array[:,1].argsort()])
                        plane_mesh = [mesh[0], 1, mesh[2]]

                    if i==2:
                        sort_indx = np.lexsort(dis_array[dis_array[:,2].argsort()])
                        plane_mesh = [mesh[0], mesh[1], 1]

            sorted_dist = dis_array(sort_index)
            sorted_energies = ebands(sort_indx)
            sorted_kpoints = self._kpoints(sort_index)

            for dist, index in enumerate(sorted_dist):
                while np.abs(dist- np.min(sorted_dist))<0.001:
                    plane_bands.append(sorted_energies[index])

            sort_idx = np.lexsort((sorted_kpoints[:, 2], sorted_kpoints[:, 1], sorted_kpoints[:, 0]))
            energies_sorted = sorted_energies[sort_idx]

            return energies.reshape(plane_mesh)




class BrillouinZone(object):

    """An object which holds information for the Brillioun Zone. This is the Wigner–Seitz cell
        of the reciprocal lattice.
    """

    def __init__(self, rlattvec: np.array):
        """Summary
        
        Args:
            lattvec (np.array): The lattice vector (b1, b2, b3) in reciprocal space.
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

def project(vector, plane):
    theta = np.array([0, 0, np.pi])
    a = np.array([[1, 0, 0],
                [0, np.cos(theta[0]), np.sin(theta[0])], 
                [0, -np.sin(theta[0]), np.cos(theta[0])]])

    b = np.array([[np.cos(theta[1]), 0, -np.sin(theta[1])],
                [0, 1, 0], 
                [np.sin(theta[1]), 0, np.cos(theta[1])]])

    c = np.array([[np.cos(theta[2]), np.sin(theta[2]), 0],
                [0, 1, 0], 
                [np.sin(theta[1]), 0, np.cos(theta[1])]])

    d = np.matmul(np.matmul(a,np.matmul(b,c)), vector)

    e = np.array([[1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

    f = np.matmul(e, np.append(d, 1))

    b_x = f[0]/f[3]
    b_y = f[1]/f[3]

    return [b_x, b_y, 0]

def plane_dist(slice_plane, vertex):

    return (np.linalg.norm(slice_plane[0]*vertex[0] + slice_plane[1]*vertex[1] + slice_plane[2]*vertex[2] + slice_plane[3]))/(np.sqrt(slice_plane[0]**2 + slice_plane[1]**2 + slice_plane[2]**2))

def compute_energy_contours(energies, contour:float):

    contours = measure.find_contours(energies, contour)

    return contours

