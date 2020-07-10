"""
Tools for computing the various weights and roots of SU(N).
"""
import warnings
import numpy as np
from scipy.special import digamma
from . import ConfinementWarning


def get_weights(N):
    """Computes the weights of the fundamental representation of SU(N).

    Parameters
    ----------
    N : int
        The degree of SU(N).

    Returns
    -------
    nu : ndarray
        Array of shape (N-1, N) containing the weights of the fundamental
        representation.
    """
    return np.stack([_nu(b, N) for b in range(1, N + 1)])


def get_fundamental_weights(N):
    """Computes the fundamental weights of SU(N).

    Parameters
    ----------
    N : int
        The degree of SU(N).

    Returns
    -------
    w : ndarray
        Array of shape (N-1, N-1) containing the fundamental weights.
    """
    return np.stack([_w(b, N) for b in range(1, N)])


def get_weyl_vector(N):
    """Computes the Weyl vector of SU(N).

    Parameters
    ----------
    N : int
        The degree of SU(N).

    Returns
    -------
    rho : ndarray
        1D array of size N-1 representing the Weyl vector.
    """
    return np.sum(get_fundamental_weights(N), axis=0)


def get_simple_roots(N):
    """Computes the simple roots (and affine root) of SU(N).

    Parameters
    ----------
    N : int
        The degree of SU(N).

    Returns
    -------
    alpha : ndarray
        Array of shape (N, N-1) contianing the simple roots. alpha[N-1] is the
        affine root, and alpha[a] for a < N-1 are the simple roots.
    """
    return np.stack([_alpha(b, N) for b in range(1, N)] + [_affine_root(N)])


def kahler_metric(N, epsilon=0.):
    """Computes the inverse Kahler metric to first order with weak coupling.

    Parameters
    ----------
    N : int
        The degree of SU(N).
    epsilon : float
        The expansion parameter, which determines the strength of the leading
        order quantum correction.

    Returns
    -------
    g : ndarray
        Array of shape (N-1, N-1) giving the inverse Kahler metric as a matrix.
    """
    g = np.identity(N - 1)
    for i in range(1, N):
        for j in range(1, N):
            for A in range(1, N + 1):
                for B in range(A + 1, N + 1):
                    factor1 = _lambda(i, A) - _lambda(i, B)
                    factor2 = _lambda(j, A) - _lambda(j, B)
                    factor3 = digamma((B - A) / N) + digamma(1 - (B - A) / N)
                    g[i - 1, j - 1] += epsilon * factor1 * factor2 * factor3

    eigenvalues = np.linalg.eigvals(g)
    if not np.all(eigenvalues > 0):
        warnings.warn("Kahler metric is not positive-definite",
                      ConfinementWarning)

    return g


def _delta(i, j):
    """The Kronecker delta. Returns 1 if i == j, 0 otherwise."""
    return int(i == j)


def _theta(a, b):
    """Returns 1 if a >= b, 0 otherwise."""
    return int(a >= b)


def _lambda(a, b):
    """The ath component of the bth weight of the funamental representation."""
    return (_theta(a, b) - a * _delta(a + 1, b)) / (a * (a + 1))**0.5


def _nu(b, N):
    """The bth weight of the fundamental representation of SU(N)."""
    return np.array([_lambda(a, b) for a in range(1, N)])


def _w(b, N):
    """The bth fundamental weight of SU(N)."""
    return sum([_nu(a, N) for a in range(1, b + 1)])


def _alpha(b, N):
    """The bth simple root of SU(N)."""
    return _nu(b, N) - _nu(b + 1, N)


def _affine_root(N):
    """The affine root of SU(N)."""
    return -sum([_alpha(b, N) for b in range(1, N)])
