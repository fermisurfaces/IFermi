"""BoltzTraP2 helper functions."""

import numpy as np

__all__ = ["bands_fft", "worker"]


def bands_fft(equiv, coeffs, lattvec, nworkers=1):
    """Rebuild the full energy bands from the interpolation coefficients.

    Adapted from BoltzTraP2.

    Args:
        equiv: list of k-point equivalence classes in direct coordinates
        coeffs: interpolation coefficients
        lattvec: lattice vectors of the system
        nworkers: number of working processes to span

    Returns:
        A 3-tuple (eband, vband) containing the energy bands  and group velocities.
        The shapes of those arrays are (nbands, nkpoints), (nbands, 3, nkpoints)
        where nkpoints is the total number of k points on the grid.
    """
    import multiprocessing as mp

    dallvec = np.vstack(equiv)
    sallvec = mp.sharedctypes.RawArray("d", dallvec.shape[0] * 3)
    allvec = np.frombuffer(sallvec)
    allvec.shape = (-1, 3)
    dims = 2 * np.max(np.abs(dallvec), axis=0) + 1
    np.matmul(dallvec, lattvec.T, out=allvec)
    eband = np.zeros((len(coeffs), np.prod(dims)))
    vband = np.zeros((len(coeffs), 3, np.prod(dims)))

    # Span as many worker processes as needed, put all the bands in the queue,
    # and let them work until all the required FFTs have been computed.
    workers = []
    iq = mp.Queue()
    oq = mp.Queue()
    for iband, bandcoeff in enumerate(coeffs):
        iq.put((iband, bandcoeff))

    # The "None"s at the end of the queue signal the workers that there are
    # no more jobs left and they must therefore exit.
    for i in range(nworkers):
        iq.put(None)

    for i in range(nworkers):
        workers.append(mp.Process(target=worker, args=(equiv, sallvec, dims, iq, oq)))

    for w in workers:
        w.start()

    # The results of the FFTs are processed as soon as they are ready.
    for r in range(len(coeffs)):
        iband, eband[iband], vband[iband] = oq.get()

    for w in workers:
        w.join()

    return eband.real, vband.transpose(0, 2, 1)


def worker(equivalences, sallvec, dims, iqueue, oqueue):
    """Thin wrapper around FFTev and FFTc to be used as a worker function.

    Adapted from BoltzTraP2.

    Args:
        equivalences: list of k-point equivalence classes in direct coordinates
        sallvec: Cartesian coordinates of all k points as a 1D vector stored
                    in shared memory.
        dims: upper bound on the dimensions of the k-point grid
        iqueue: input multiprocessing.Queue used to read bad indices
            and coefficients.
        oqueue: output multiprocessing.Queue where all results of the
            interpolation are put. Each element of the queue is a 4-tuple
            of the form (index, eband, vvband, cband), containing the band
            index, the energies, and the group velocities.

    Returns:
        None. The results of the calculation are put in oqueue.
    """
    from BoltzTraP2.fite import FFTev

    allvec = np.frombuffer(sallvec)
    allvec.shape = (-1, 3)

    while True:
        task = iqueue.get()
        if task is None:
            break
        else:
            index, bandcoeff = task

        eband, vband = FFTev(equivalences, bandcoeff, allvec, dims)
        oqueue.put((index, eband, vband))
