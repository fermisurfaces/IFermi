from interpolator import Interpolater
from pymatgen.io.ase import AseAtomsAdaptor

from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure


import numpy as np

import matplotlib.pyplot as plt

import math

from skimage import measure

def plane_dist(slice_plane, vertex):

    return (np.linalg.norm(slice_plane[0]*vertex[0] + slice_plane[1]*vertex[1] + slice_plane[2]*vertex[2] + slice_plane[3]))/(np.sqrt(slice_plane[0]**2 + slice_plane[1]**2 + slice_plane[2]**2))

def compute_energy_contours(energies, contour:float):

    contours = measure.find_contours(energies, contour)

    return contours



vr = Vasprun("/home/amy/Documents/ferm-surf/data_Bi2Te3/vasprun.xml")
    
bs = vr.get_band_structure()

interpolater = Interpolater(bs) 

new_bs, hdims = interpolater.interpolate_bands(1)

slice_array = [0,0,1]

kpoints = np.array([k.frac_coords for k in new_bs.kpoints])

contour = 0

# mesh = [np.unique([i[0] for i in kpoints]).size, np.unique([i[1] for i in kpoints]).size, np.unique([i[2] for i in kpoints]).size]
mesh = 2 * hdims + 1

final_energies = []

mu = 0.02


print(mesh)

for spin in new_bs.bands.keys():

    ebands = new_bs.bands[spin]

    ebands -= new_bs.efermi -mu

    plane_bands = []

    sorted_energies = []

    new_sorted_energies = []

    dis_array = np.array([plane_dist(np.append(np.array(slice_array), 0.0), i) for i in kpoints])

    for i, value in enumerate(slice_array):
        if not value == 0:
            if i==0:
                sortd_indx = np.argsort(dis_array)
                plane_mesh = [mesh[2], mesh[1]]
            if i==1:
                sortd_indx = np.argsort(dis_array)
                plane_mesh = [mesh[2], mesh[0]]

            if i==2:
                sortd_indx = np.argsort(dis_array)
                plane_mesh = [mesh[0], mesh[1]]

    sorted_dist = dis_array[sortd_indx]
    for band in ebands:
        sorted_energies.append(band[sortd_indx])
    sorted_kpoints = kpoints[sortd_indx]

    for band in sorted_energies:
        final_kpts = []
        plane_bands = []
        for index, dist in enumerate(sorted_dist):
            if np.abs(dist- np.min(sorted_dist))<0.001:
                plane_bands.append(band[index])
                final_kpts.append(sorted_kpoints[index])
        new_sorted_energies.append(np.array(plane_bands))


    final_kpts = np.array(final_kpts)

    sort_idx = np.lexsort((final_kpts[:, 2], final_kpts[:, 2]<0,
                                final_kpts[:, 1], final_kpts[:, 1]<0,
                                final_kpts[:, 0], final_kpts[:, 0]<0))

    
    for band in new_sorted_energies:

        print(len(sort_idx))

        print(len(band))

        energies_sorted = band[sort_idx]

        final_energies.append(energies_sorted.reshape(plane_mesh))

    fig, ax = plt.subplots()

    for energy in final_energies:

        contours = measure.find_contours(energy, 0.0)

    # Display the image and plot all contours found

        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(r"slice of FS at kz = 0 for \mu = %.3f" % (mu))
    plt.show()




