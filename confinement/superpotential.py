import numpy as np
from scipy.integrate import simps
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
        r"""Evaluate this Superpotential on a field.

        Parameters
        ----------
        field : Field or ndarray
            The field on which to evaluate. This vector field must have N-1
            component scalar fields. If type is ndarray, then the array must
            have shape (N-1,).

        Returns
        -------
        W : ndarray or float
            The value of the superpotential at each point. If field.field has
            shape (N-1, nz, ny), then W has shape (nz, ny). If field.field has
            shape (N-1, nz), then W has shape (nz,). If field is an ndarray,
            with shape (N-1,), then W is a scalar.

        Notes
        -----
        The superpotential `W` evaluated on a vector field
        :math:`\boldsymbol{x}` is given by

        .. math::
            W(\boldsymbol{x}) = \sum_{a = 1}^{N} e^{\boldsymbol{\alpha}_a
            \cdot \boldsymbol{x}}

        where :math:`\boldsymbol{\alpha}_a`, :math:`a = 1, \ldots, N - 1` are
        the simple roots of SU(`N`), and :math:`\boldsymbol{\alpha}_N` is the
        affine root.
        """
        if isinstance(field, np.ndarray):
            dot_products = np.sum(self.alpha * field[np.newaxis, :], axis=1)
        elif field.field.ndim == 3:
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
            The potential energy density at each point. If field.field has shape
            (N-1, nz, ny), then energy_density has shape (nz, ny). Likewise,
            if field.field has shape (N-1, nz), then energy_density has shape
            (nz,).
        """
        return np.sum(np.abs(self.gradient(field))**2 / 4, axis=0)

    def energy(self, field):
        """Compute the energy of a field under this Superpotential.

        Parameters
        ----------
        field : Field
            The field on which to evaluate. This vector field must have N-1
            component scalar fields.

        Returns
        -------
        energy : float
            The total potential energy.
        """
        # Compute the energy density and repeatedly integrate over all axes
        density = self.energy_density(field)
        return _integrate_energy_density(density, field)

    def total_energy_density(self, field, **kwargs):
        """Compute the total energy of a field under this Superpotential.

        Parameters
        ----------
        field : Field
            The field on which to evaluate. This vector field must have N-1
            component scalar fields.
        **kwargs
            Keyword arguments to pass to field.energy_density().

        Returns
        -------
        energy_density : ndarray
            The total energy density at each point. If field.field has shape
            (N-1, nz, ny), then energy_density has shape (nz, ny). Likewise,
            if field.field has shape (N-1, nz), then energy_density has shape
            (nz,).
        """
        return self.energy_density(field) + field.energy_density(**kwargs)

    def total_energy(self, field, **kwargs):
        """Compute the total energy of a field under this Superpotential.

        Parameters
        ----------
        field : Field
            The field on which to evaluate. This vector field must have N-1
            component scalar fields.
        **kwargs
            Keyword arguments to pass to field.energy_density().

        Returns
        -------
        energy : float
            The total energy.
        """
        # Compute the energy density and repeatedly integrate over all axes
        density = self.total_energy_density(field, **kwargs)
        return _integrate_energy_density(density, field)

    def eom(self, field):
        r"""Compute the equation of motion term due to this Superpotential.

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

        Notes
        -----
        The potential term in the equation of motion is given by

        .. math::
            \frac{1}{4} \frac{\partial}{\partial (\boldsymbol{x}^*)}
            \left| \frac{dW}{d \boldsymbol{x}} \right|^2.
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
        r"""Compute the first derivative of a field from the BPS equation.

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

        Notes
        -----
        The BPS equation is given by

        .. math::
            \frac{d \boldsymbol{x}}{dz} = \frac{\alpha}{2}
            \frac{dW^*}{d \boldsymbol{x}^*}

        where

        .. math::
            \alpha =
            \frac{W(\boldsymbol{x}(\infty)) - W(\boldsymbol{x}(-\infty))}
            {|W(\boldsymbol{x}(\infty)) - W(\boldsymbol{x}(-\infty))|}.
        """
        # Values of the Superpotential at +/- infinity
        left_val, right_val = self(field)[[0, -1]]
        factor = (right_val - left_val) / np.abs(right_val - left_val)

        # Compute the gradient and return the BPS derivative
        gradient = self.gradient(field)
        return np.conj(gradient) * factor / 2

    def bps_eom(self, field):
        r"""Compute the second-order BPS equation of motion for a 1D field.

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

        Notes
        -----
        The second-order BPS equation is given by

        .. math::
            \frac{d^2 x_i}{dz^2} = \frac{1}{4} \sum_{j = 1}^{N - 1}
            \frac{\partial W}{\partial x_j}
            \frac{\partial^2 W^*}{\partial x_j^* \partial x_i^*}.
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

    def bps_energy(self, vacuum1, vacuum2):
        r"""Compute the energy of a BPS soliton interpolating between two vacua.

        Parameters
        ----------
        vacuum1 : ndarray
            Array of shape (N-1,) giving the vacuum at negative infinity.
        vacuum2 : ndarray
            Array of shape (N-1,) giving the vacuum and positive infintiy.

        Returns
        -------
        bps_energy : float
            The energy of a BPS soliton interpolating between the two vacua.

        Notes
        -----
        The BPS soliton energy is given by :math:`|W(\infty) - W(-\infty)|`.
        """
        return np.abs(self(vacuum2) - self(vacuum1))

    def bps_energy_exact(self, k):
        r"""Compute the energy of a BPS k-wall interpolating between two vacua.

        Parameters
        ----------
        k : int
            The number of units separating the vacua.

        Returns
        -------
        bps_energy : float
            The energy of a BPS soliton interpolating between the two vacua.

        Notes
        -----
        This method serves the same purpose as Superpotential.bps_energy(), but
        uses the exact expression for the BPS k-wall energy given by

        .. math::
            E_\mathrm{BPS}^{k\mathrm{-wall}} =
            N \sqrt{2 ( 1 - \cos( 2 \pi k / N ) )}.
        """
        return self.N * (2 * (1 - np.cos(2 * np.pi * k / self.N)))**0.5

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


def _integrate_energy_density(density, field):
    """Integrate an energy density over a grid.

    Parameters
    ----------
    density : ndarray
        An array giving the energy density at each point.
    field : Field
        The Field which defines the grid. The grid should have the same shape
        as the density array.

    Returns
    -------
    energy : float
        The integrated energy.
    """
    if density.ndim == 2:
        return simps(simps(density, x=field.y), x=field.z)
    elif density.ndim == 1:
        return simps(density, x=field.z)
    else:
        raise ValueError("field has incorrect shape")
