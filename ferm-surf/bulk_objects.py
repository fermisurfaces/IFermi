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
    
    def __init__(self, bs: BandStructure, hdims,
                 soc: bool = False, doping = None) -> None:
        """
        Args:
            k_dim (int): Description
            fermi_energy (int): The Fermi energy
            iso_surface (np.array): A List containing all parameters for each surface at the Fermi surface. 
            n_bands (int): Number of bands which cross the Fermi-Surface
            is_spin_polarised (bool, optional): set to True to perform a spin-polarised.
            The index i will access [verts, faces, normals, values] for surface i. 
        """
        self._fermi_energy = bs.efermi

        self._kpoints = np.array([k.frac_coords for k in bs.kpoints])
        
        self._hdims = hdims

        dims = 2 * hdims + 1

        self._dims = dims

        self._k_dim = (dims[0], dims[1], dims[2])

        # self.rearrange_bands(bs)

        # self.trim_energies(bs)

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
    def fermi_energy(self):
        return self._fermi_energy

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
        """Use ski-kit's marching cubes algorithm to compute 
        the isosurfaces of the energy bands. (surfaces correpond to points
        in k-space where E = E_fermi)
        
        Args:
            bs (BandStructure): Pymatgen BandStructure Object
        """
        total_data = []
        iso_surface = []
        n_bands = 0

        for spin in bs.bands.keys():

            ebands = bs.bands[spin]
            ebands -= bs.efermi
            emax = ebands.max(axis=1)
            emin = ebands.min(axis=1)


            for  band in ebands:
                n_bands += 1
                i, j, k = 0, 0, 0
                data_min = 0
                data_max = 0
                data = np.zeros((self._k_dim[0], self._k_dim[1], self._k_dim[2]))
                for energy in band:
                
                    data[i][j][k] = energy
                    if energy<data_min or data_min ==0:
                        data_min = energy
                    if energy>data_max or data_max==0:
                        data_max = energy
                            
                    if j == (self._k_dim[1] - 1) and k == (self._k_dim[2] - 1):
                        j = 0
                        k = 0
                        i += 1
                    
                    elif k == (self._k_dim[2] - 1):
                        j += 1
                        k = 0
                    
                    else:
                        k += 1

                if 0 > data_min and 0 < data_max:
                    total_data.append(data)

        rlattvec = bs.lattice_rec._matrix


        for band_index, band_data in enumerate(total_data):

            verts, faces, normals, values = measure.marching_cubes_lewiner(band_data, 0,
                                                                           self._spacing)
            iso_surface.append([verts, faces, normals, values])


        self._iso_surface = iso_surface

    def rearrange_bands(self, bs):
        """Changes the order of the bands so that the surface from (-0.5*b_i, 0.5*b_i) is plotted
        where b_i is the reciprocal lattice vector in directions {1, 2, 3}.
        
        Args:
            bs (BandStructure): pymatgen BandStrucutre object whose bands are to be rearranged.
        """
        for spin in bs.bands.keys():
            ebands = bs.bands[spin]

            dims = self._dims
            hdims = self._hdims

            new_ebands = []

            for band in ebands:
                
                old_bands = band
                band = np.concatenate((old_bands[int((len(old_bands) + dims[2] * dims[1]) / 2):],
                                          old_bands[:int((len(old_bands) + dims[2] * dims[1]) / 2)]), axis=0)

                q = 0
                old_bands = band
                while (q + 1) * dims[1] * dims[2] < len(old_bands) - 1:
                    sub_array = old_bands[q * dims[2] * dims[1]:(q + 1) * dims[2] * dims[1]:1]
                    band = np.concatenate((band[:(q) * dims[2] * dims[1]], sub_array[(hdims[1] + 1) * dims[2]:],
                                              sub_array[:(hdims[1] + 1) * dims[2] + 1],
                                           band[(q) * dims[2] * dims[1] + dims[2] * dims[1] + 1:]), axis=0)

                    q += 1

                old_bands = band
                q = 0
                while (q + 1) * dims[2] < len(old_bands) - 1:
                    sub_array = old_bands[q * dims[2]:(q + 1) * dims[2]:1]
                    new = (np.concatenate((sub_array[(hdims[2] + 1):], sub_array[:(hdims[2] + 1)]), axis=0))
                    band = np.concatenate((band[:(q) * dims[2]], sub_array[(hdims[2] + 1):], sub_array[:(hdims[2] + 2)],
                                           band[(q) * dims[2] + dims[2] + 1:]), axis=0)
                    q += 1

                new_ebands.append(band)

            bs.bands[spin] = np.array(new_ebands)



    def trim_energies(self, bs):
        """Sets energies not in the Brillouin Zone to None. 
        
        Args:
            bs (BandStructure): BandStrucutre object
        """
        bz = BrillouinZone(bs.structure.lattice.reciprocal_lattice)

        for spin in bs.bands.keys():
            ebands = bs.bands[spin]

            emax = ebands.max(axis=1)
            emin = ebands.min(axis=1)

            #Remove all points outside the BZ
            for i, ijk0 in enumerate(itertools.product(*(range(0, d) for d in self._dims))):
                ijk = [
                    ijk0[j] if ijk0[j] <= self._hdims[j] else ijk0[j] - self._dims[j]
                    for j in range(len(self._dims))
                ]
                abc = np.array(ijk, dtype=np.float64) / np.array(self._dims)
                xyz = bs.lattice_rec._matrix @ abc
                for c, n in zip(bz._centers, bz._normals):
                    if np.dot(xyz - c, n) > 0.:
                        ebands[:, i] = None
                        break



class BrillouinZone(object):

    """An object which holds information for the Brillioun Zone. This is the Wigner–Seitz cell
        of the reciprocal lattice. Not currently used in the plotter. Once I have included a mthod which 
        reloacted the bands to the Brillouin Zone, I will update this.
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

        to_return = []
        for r in voronoi.ridge_dict:
            if r[0] == 13 or r[1] == 13:
                to_return.append([voronoi.vertices[i] for i in voronoi.ridge_dict[r]])

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
        self._to_return = to_return

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

