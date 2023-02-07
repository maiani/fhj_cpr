import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.optimize import linear_sum_assignment

def sort_eigensystem(ws : np.ndarray, vs : np.ndarray) -> (np.ndarray, np.ndarray):
    """Sort the eigensystem of an Hamiltonian.
    
    Parameters:
    -----------
    ws : eigenvalues
    vs : eigenvectos
    
    Returns:
    ws_sorted : sorted eigenvalues
    vs_sorted : sorted eigenvectors
    """
    
    def best_match(psi1, psi2, threshold=None):
        """Find the best match of two sets of eigenvectors.

        Parameters:
        -----------
        psi1, psi2 : numpy 2D complex arrays
            Arrays of initial and final eigenvectors.
        threshold : float, optional
            Minimal overlap when the eigenvectors are considered belonging to the same band.
            The default value is :math:`1/(2N)^{1/4}`, where :math:`N` is the length of each eigenvector.

        Returns:
        --------
        sorting : numpy 1D integer array
            Permutation to apply to ``psi2`` to make the optimal match.
        diconnects : numpy 1D bool array
            The levels with overlap below the ``threshold`` that should be considered disconnected.
        """
        if threshold is None:
            threshold = (2 * psi1.shape[0]) ** -0.25
        Q = np.abs(psi1.T.conj() @ psi2)  # Overlap matrix
        orig, perm = linear_sum_assignment(-Q)
        return perm, Q[orig, perm] < threshold

    N = ws.shape[0]

    e = ws[0]
    psi = vs[0]

    ws_sorted = [e]
    vs_sorted = [psi]

    for i in range(1, N):
        e2 = ws[i]
        psi2 = vs[i]
        perm, line_breaks = best_match(psi, psi2)
        e2 = e2[perm]
        intermediate = (e + e2) / 2
        intermediate[line_breaks] = None
        psi = psi2[:, perm]
        e = e2

        ws_sorted.append(intermediate)
        ws_sorted.append(e)
        vs_sorted.append(psi)

    return np.array(ws_sorted)[::2], np.array(vs_sorted)