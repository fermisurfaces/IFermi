"""k-point manipulation functions."""

import warnings
from typing import Tuple

import numpy as np
from pymatgen.electronic_structure.bandstructure import BandStructure

from ifermi.defaults import KTOL

__all__ = [
    "kpoints_to_first_bz",
    "kpoints_from_bandstructure",
    "get_kpoint_mesh_dim",
    "get_kpoint_spacing",
    "sort_boltztrap_to_spglib",
]


def kpoints_to_first_bz(kpoints: np.ndarray, tol: float = KTOL) -> np.ndarray:
    """Translate fractional k-points to the first Brillouin zone.

    I.e. all k-points will have fractional coordinates:
        -0.5 <= fractional coordinates < 0.5

    Args:
        kpoints: A (n, 3) float array of the k-points in fractional coordinates.
        tol: Tolerance for treating two k-points as equivalent.

    Returns:
        A (n, 3) float array of the translated k-points.
    """
    kp = kpoints - np.round(kpoints)

    # account for small rounding errors for 0.5
    round_dp = int(np.log10(1 / tol))
    krounded = np.round(kp, round_dp)

    kp[krounded == -0.5] = 0.5
    return kp


def get_kpoint_mesh_dim(kpoints: np.ndarray, tol: float = KTOL) -> Tuple[int, int, int]:
    """
    Get the k-point mesh dimensions.

    Args:
        kpoints: A (n, 3) float array of the k-points in fractional coordinates.
        tol: Tolerance for treating two k-points as equivalent.

    Returns:
        A (3, ) int array of the k-point mesh dimensions.
    """
    round_dp = int(np.log10(1 / tol))
    round_kpoints = np.round(kpoints, round_dp)

    nx = len(np.unique(round_kpoints[:, 0]))
    ny = len(np.unique(round_kpoints[:, 1]))
    nz = len(np.unique(round_kpoints[:, 2]))

    return nx, ny, nz


def sort_boltztrap_to_spglib(kpoints: np.ndarray) -> np.ndarray:
    """
    Get an index array that sorts the k-points from BoltzTraP2 to the order from spglib.

    Args:
        kpoints: A (n, 3) float array of the k-points in fractional coordinates.

    Returns:
        A (n, ) int array of the sort order.
    """
    sort_idx = np.lexsort(
        (
            kpoints[:, 2],
            kpoints[:, 2] < 0,
            kpoints[:, 1],
            kpoints[:, 1] < 0,
            kpoints[:, 0],
            kpoints[:, 0] < 0,
        )
    )
    boltztrap_kpoints = kpoints[sort_idx]

    sort_idx = np.lexsort(
        (
            boltztrap_kpoints[:, 0],
            boltztrap_kpoints[:, 0] < 0,
            boltztrap_kpoints[:, 1],
            boltztrap_kpoints[:, 1] < 0,
            boltztrap_kpoints[:, 2],
            boltztrap_kpoints[:, 2] < 0,
        )
    )
    return sort_idx


def get_kpoint_spacing(kpoints: np.ndarray) -> np.ndarray:
    """Get the spacing between fractional k-points.

    Args:
        kpoints: A (n, 3) float array of the k-points in fractional coordinates.

    Returns:
        A (3, ) float array of the spacing along each reciprocal lattice direction.
    """
    kpoints = kpoints.round(8)
    unique_a = np.unique(kpoints[:, 0])
    unique_b = np.unique(kpoints[:, 1])
    unique_c = np.unique(kpoints[:, 2])

    diff_a = np.diff(unique_a)
    diff_b = np.diff(unique_b)
    diff_c = np.diff(unique_c)

    if not (
        np.allclose(diff_a - diff_a[0], 0, atol=1e-7)
        and np.allclose(diff_b - diff_b[0], 0, atol=1e-7)
        and np.allclose(diff_c - diff_c[0], 0, atol=1e-7)
    ):
        warnings.warn("k-point mesh is not uniform")

    return np.array([diff_a[0], diff_b[0], diff_c[0]])


def kpoints_from_bandstructure(
    bandstructure: BandStructure, cartesian: bool = False
) -> np.ndarray:
    """
    Extract the k-points from a band structure.

    Args:
        bandstructure: A band structure object.
        cartesian: Whether to return the k-points in cartesian coordinates.

    Returns:
        A (n, 3) float array of the k-points.
    """
    if cartesian:
        kpoints = np.array([k.cart_coords for k in bandstructure.kpoints])
    else:
        kpoints = np.array([k.frac_coords for k in bandstructure.kpoints])

    return kpoints
