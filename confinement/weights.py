"""
Tools for computing the various weights and roots of SU(N).
"""
import numpy as np


def get_weights(N):
    """Computes the weights of the fundamental representation of SU(N).

    Parameters
    ----------
    N : int
        The degree of SU(N).

    Returns
    -------
    nu : ndarray
        Array of shape (N-1, N-1) containing the weights of the fundamental
        representation.
    """
    return np.stack([_nu(b, N) for b in range(1, N)])


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
        Array of shape (N, N-1) contianing the simple roots. alpha[0] is the
        affine root, and alpha[a] for a > 0 are the simple roots.
    """
    return np.stack([_affine_root(N)] + [_alpha(b, N) for b in range(1, N)])


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
