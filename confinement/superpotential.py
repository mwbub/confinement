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
        self.alpha_shifted = np.roll(self.alpha, -1, axis=0)

    def __call__(self, field):
        """Evaluate this Superpotential on a field.

        Parameters
        ----------
        field : Field
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

    def field_laplacian(self, field):
        """Compute the Laplacian term on a field due to this Superpotential.

        Parameters
        ----------
        field : Field
            The field on which to evaluate. This vector field must have N-1
            component scalar fields.

        Returns
        -------
        laplacian : ndarray
            Array giving the value of the Laplacian due to this Superpotential
            at each point. Has the same shape as field.field.
        """
        f = field.field

        # Add new axes to the root arrays for vectorized operations
        alpha = self.alpha[:, :, np.newaxis, np.newaxis]
        alpha_shifted = self.alpha_shifted[:, :, np.newaxis, np.newaxis]

        # Compute the dot product of the field with the roots, and add an axis
        dot_products = _dot_roots_with_field(self.alpha, f)[:, np.newaxis, :, :]

        # Exponentiate the dot products, shift left, and compute the conjugates
        exp = np.exp(dot_products)
        exp_shifted = np.roll(exp, -1, axis=0)
        exp_conj = np.conj(exp)
        exp_conj_shifted = np.conj(exp_shifted)

        # Compute the terms of the summand
        term1 = alpha * exp * exp_conj
        term2 = alpha * exp_shifted * exp_conj
        term3 = alpha_shifted * exp * exp_conj_shifted
        summand = 2 * term1 - term2 - term3

        # Return the sum
        return np.sum(summand, axis=0)

    def _field_laplacian_naive(self, field):
        """Naive implementation of field_laplacian, used for testing purposes.

        Parameters
        ----------
        field : Field
            The field on which to evaluate. This vector field must have N-1
            component scalar fields.

        Returns
        -------
        laplacian : ndarray
            Array giving the value of the Laplacian due to this Superpotential
            at each point. Has the same shape as field.field.
        """
        f = field.field
        alpha = self.alpha[:, :, np.newaxis, np.newaxis]
        laplacian = np.zeros_like(f)

        dot_products = _dot_roots_with_field(self.alpha, f)[:, np.newaxis, :, :]
        exp = np.exp(dot_products)
        exp_conj = np.conj(exp)

        for a in range(self.N):
            for b in range(self.N):
                laplacian += (alpha[b] * np.sum(alpha[a] * alpha[b]) * exp[a]
                              * exp_conj[b])
        return laplacian


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
