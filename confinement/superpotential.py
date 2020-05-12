import numpy as np
from .weights import get_simple_roots


class Superpotential:
    """
    A class representing the superpotential for a super Yang-Mills theory
    compactified on R^3 x S^1 in the small circle limit.
    """

    def __init__(self, N):
        """Initialize this Superpotential.

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
        field : Field
            The field on which to evaluate. This vector field must have N-1
            component scalar fields.

        Returns
        -------
        W : ndarray
            The value of the superpotential at each point. If field.field has
            shape (N-1, nz, ny), then W has shape (nz, ny). If field.field has
            shape (N-1, nz), then W has shape (nz,).
        """
        if field.field.ndim == 3:
            dot_products = _dot_roots_with_field2d(self.alpha, field.field)
        elif field.field.ndim == 2:
            dot_products = _dot_roots_with_field1d(self.alpha, field.field)
        else:
            raise ValueError("field has incorrect shape")
        return np.sum(np.exp(dot_products), axis=0)

    def gradient(self, field):
        """Compute the gradient of this Superpotential with respect to a field.

        Parameters
        ----------
        field : Field
            The field on which to evaluate. This vector field must have N-1
            component scalar fields.

        Returns
        -------
        gradient : ndarray
            The gradient of this Superpotential at each point. Has the same
            shape as field.field.
        """
        if field.field.ndim == 3:
            dot_products = _dot_roots_with_field2d(self.alpha, field.field)
            exp = np.exp(dot_products)[:, np.newaxis, :, :]
            summand = exp * self.alpha[:, :, np.newaxis, np.newaxis]
        elif field.field.ndim == 2:
            dot_products = _dot_roots_with_field1d(self.alpha, field.field)
            exp = np.exp(dot_products)[:, np.newaxis, :]
            summand = exp * self.alpha[:, :, np.newaxis]
        else:
            raise ValueError("field has incorrect shape")

        return np.sum(summand, axis=0)

    def energy_density(self, field):
        """Compute the energy density of a field under this Superpotential.

        Parameters
        ----------
        field : Field
            The field on which to evaluate. This vector field must have N-1
            component scalar fields.

        Returns
        -------
        energy_density : ndarray
            The energy density at each point. If field.field has shape
            (N-1, nz, ny), then energy_density has shape (nz, ny). Likewise,
            if field.field has shape (N-1, nz), then energy_density has shape
            (nz,).
        """
        # Compute the gradients of the field and the potential
        if field.field.ndim == 3:
            df_dz, df_dy = field.gradient()
        elif field.field.ndim == 2:
            df_dz = field.gradient()
            df_dy = np.zeros_like(df_dz)
        else:
            raise ValueError("field has incorrect shape")
        dw = self.gradient(field)

        # Compute the energy density
        summand = np.abs(df_dz)**2 + np.abs(df_dy)**2 + np.abs(dw)**2 / 4
        return np.sum(summand, axis=0)

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
        dot_products = _dot_roots_with_field2d(self.alpha, field.field)

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

    def bps(self, field):
        """Compute the first derivative of a field from the BPS equation.

        Parameters
        ----------
        field : Field1D
            The field on which to evaluate. This vector field must have N-1
            component scalar fields. The leftmost and rightmost values of the
            field are assumed to be the desired boundary values at infinity.

        Returns
        -------
        df : ndarray
            Array giving the value of the derivative of the field under the BPS
            equation at each point. Has the same shape as field.field.
        """
        # Values of the Superpotential at +/- infinity
        left_val, right_val = self(field)[[1, -1]]
        factor = (right_val - left_val) / np.abs(right_val - left_val)

        # Compute the gradient and return the BPS derivative
        gradient = self.gradient(field)
        return np.conj(gradient) * factor / 2

    def bps_eom(self, field):
        """Compute the second-order BPS equation of motion for a 1D field.

        Parameters
        ----------
        field : Field1D
            The field on which to evaluate. This vector field must have N-1
            component scalar fields.

        Returns
        -------
        deriv : ndarray
            Array giving the value of the second derivative of the field under
            the BPS equation at each point. Has the same shape as field.field.
        """
        # Compute the dot products of the field with the roots
        dot_products = _dot_roots_with_field1d(self.alpha, field.field)

        # Exponentiate the dot products and compute the conjugate
        exp = np.exp(dot_products)
        exp_conj = np.conj(exp)

        # Compute the first inner sum
        sum1 = exp[:, np.newaxis, :] * self.alpha[:, :, np.newaxis]
        sum1 = np.sum(sum1, axis=0)

        # Compute the second inner sum
        sum2 = (exp_conj[:, np.newaxis, np.newaxis, :]
                * self.alpha[:, :, np.newaxis, np.newaxis]
                * self.alpha[:, np.newaxis, :, np.newaxis])
        sum2 = np.sum(sum2, axis=0)

        # Compute the outer sum
        sum3 = sum1[:, np.newaxis, :] * sum2
        sum3 = np.sum(sum3, axis=0)

        return sum3 / 4

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
        dot_products = _dot_roots_with_field2d(self.alpha, field.field)

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


def _dot_roots_with_field2d(alpha, field):
    """Compute the dot product of a 2D field with the simple roots of SU(N).

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


def _dot_roots_with_field1d(alpha, field):
    """Compute the dot product of a 1D field with the simple roots of SU(N).

    Parameters
    ----------
    alpha : ndarray
        Array of shape (N, N-1) giving the simple roots and affine root of
        SU(N), such as returned by weights.get_simple_roots(N).
    field : ndarray
        Array of shape (N-1, nz) representing the field at each point of the
        grid.

    Returns
    -------
    dot_products : ndarray
        Array of shape (N, nz) giving the dot product at each point of the grid
        for each root. The first axis represents the roots.
    """
    product = alpha[:, :, np.newaxis] * field[np.newaxis, :, :]
    return np.sum(product, axis=1)
