import numpy as np
from .weights import get_simple_roots


class Superpotential:
    """
    A class representing the superpotential for a super Yang-Mills theory
    compactified on R^3 x S^1 in the small circle limit.
    """

    def __init__(self, N):
        """Initialize this superpotential.

        Parameters
        ----------
        N : int
            The degree of SU(N).
        """
        self.N = N
        self.alpha = get_simple_roots(N)

    def __call__(self, field):
        """Evaluate this Superpotential on a field.

        Parameters
        ----------
        field : Field2D
            The field on which to evaluate. This vector field must have N-1
            component scalar fields.

        Returns
        -------
        W : ndarray
            The value of the superpotential at each point. If field.field has
            shape (N-1, nz, ny), then W has shape (nz, ny).
        """
        dot_products = _dot_roots_with_field(self.alpha, field.field)
        return np.sum(np.exp(dot_products), axis=0)

    def eom(self, field):
        """Compute the field equation of motion term due to this Superpotential.

        Parameters
        ----------
        field : Field2D
            The field on which to evaluate. This vector field must have N-1
            component scalar fields.

        Returns
        -------
        laplacian : ndarray
            Array giving the value of the Laplacian due to this Superpotential
            at each point. Has the same shape as field.field.
        """
        # Compute the dot products of the field with the roots
        dot_products = _dot_roots_with_field(self.alpha, field.field)

        # Exponentiate the dot products and add an axis for vectorized math
        exp = np.exp(dot_products)[:, np.newaxis, :, :]

        # Compute the conjugate and shifted exponential arrays
        exp_conj = np.conj(exp)
        exp_shifted_up = np.roll(exp, -1, axis=0)
        exp_shifted_down = np.roll(exp, 1, axis=0)

        # Compute the summands using vectorized operations
        summand = (self.alpha[:, :, np.newaxis, np.newaxis] * exp_conj
                   * (2 * exp - exp_shifted_up - exp_shifted_down))

        # Return the potential term of the Laplacian
        return np.sum(summand, axis=0) / 4

    def _eom_naive(self, field):
        """Naive implementation of eom, used for testing purposes.

        Parameters
        ----------
        field : Field2D
            The field on which to evaluate. This vector field must have N-1
            component scalar fields.

        Returns
        -------
        laplacian : ndarray
            Array giving the value of the Laplacian due to this Superpotential
            at each point. Has the same shape as field.field.
        """
        # Compute the dot products of the field with the roots
        dot_products = _dot_roots_with_field(self.alpha, field.field)

        # Exponentiate the dot products and add an axis for vectorized math
        exp = np.exp(dot_products)[:, np.newaxis, :, :]
        exp_conj = np.conj(exp)

        # Compute the potential term of the Laplacian by an explicit loop
        laplacian = np.zeros_like(field.field)
        for a in range(self.N):
            for b in range(self.N):
                laplacian += (self.alpha[b][:, np.newaxis, np.newaxis]
                              * np.dot(self.alpha[a], self.alpha[b])
                              * exp[a] * exp_conj[b])

        # Return the potential term of the Laplacian
        return laplacian / 4


def _dot_roots_with_field(alpha, field):
    """Compute the dot product of a field with the simple roots of SU(N).

    Parameters
    ----------
    alpha : ndarray
        Array of shape (N, N-1) giving the simple roots and affine root of
        SU(N), such as returned by weights.get_simple_roots(N).
    field : ndarray
        Array of shape (N-1, nz, ny) representing the field at each point of the
        grid.

    Returns
    -------
    dot_products : ndarray
        Array of shape (N, nz, ny) giving the dot product at each point of the
        grid for each root. The first axis represents the roots.
    """
    product = alpha[:, :, np.newaxis, np.newaxis] * field[np.newaxis, :, :, :]
    return np.sum(product, axis=1)
