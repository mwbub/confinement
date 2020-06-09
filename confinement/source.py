"""
Tools for computing the source terms for quarks.
"""
import numpy as np


class Source:
    """
    Class representing a source of electric flux.

    Attributes
    ----------
    z : float
        z coordinate of this Source.
    y : float
        y coordinate of this Source.
    """

    def __init__(self, z, y, charge, monodromy='above'):
        """Initialize this Source.

        Parameters
        ----------
        z : float
            z coordinate of this Source.
        y : float
            y coordinate of this Source.
        charge : array_like
            1D array giving the charge of the source. For SU(N), this array
            should have shape (N-1,).
        monodromy : str
            Whether the monodromy (or 'jump') due to this Source should occur
            above or below the source with respect to the y coordinate. Either
            'above' or 'below'.
        """
        self.z = z
        self.y = y
        self._charge = np.array(charge)

        if monodromy == 'above':
            self._monodromy = 1
        elif monodromy == 'below':
            self._monodromy = -1
        else:
            raise ValueError("monodromy must be either 'above' or 'below'")

    def eom(self, field):
        """Compute the field equation of motion due to this Source.

        Parameters
        ----------
        field : Field2D
            The field on which to evaluate. For SU(N), this vector field should
            have N-1 component scalar fields.

        Returns
        -------
        laplacian : ndarray
            Array giving the value of the Laplacian due to this Source at each
            point. Has the same shape as field.field.
        """
        field_val = self._charge * 2j * np.pi / field.gridsize**2
        z_index = int(round((self.z - field.zmin) / field.gridsize))
        y0 = self.y * self._monodromy

        laplacian = np.zeros_like(field.field)
        for i in range(field.ny):
            y = field.y[i] * self._monodromy
            if y >= y0:
                laplacian[:, z_index - 1, i] = field_val
                laplacian[:, z_index, i] = -field_val

        return laplacian * self._monodromy


class Meson:
    """
    Class representing an equally and oppositely charged quark-antiquark pair.

    Attributes
    ----------
    z1 : float
        Position of the quark along the z-axis.
    z2 : float
        Position of the antiquark along the z-axis.
    """

    def __init__(self, z1, z2, charge):
        """Initialize this Meson.

        Parameters
        ----------
        z1 : float
            Position of the quark along the z-axis. Should obey z1 < z2.
        z2 : float
            Position of the antiquark along the z-axis. Should obey z1 < z2.
        charge : array_like
            1D array giving the charge of the quark. For SU(N), this array
            should have shape (N-1,).
        """
        self.z1 = z1
        self.z2 = z2
        self._charge = charge

    def eom(self, field):
        """Compute the field equation of motion due to this Meson.

        Parameters
        ----------
        field : Field2D
            The field on which to evaluate. For SU(N), this vector field should
            have N-1 component scalar fields.

        Returns
        -------
        laplacian : ndarray
            Array giving the value of the Laplacian due to this Meson at each
            point. Has the same shape as field.field.
        """
        field_val = self._charge * 2j * np.pi / field.gridsize ** 2
        y_index = int(round(-field.ymin / field.gridsize))

        laplacian = np.zeros_like(field.field)
        for i in range(field.nz):
            z = field.z[i]
            if self.z1 <= z <= self.z2:
                laplacian[:, i, y_index - 1] = -field_val
                laplacian[:, i, y_index] = field_val

        return laplacian
