"""
Tools for computing the source terms for quarks.
"""
import numpy as np


class Source:
    """
    Class representing a source of electric flux.
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
        self.charge = np.array(charge)

        if monodromy == 'above':
            self.monodromy = 1
        elif monodromy == 'below':
            self.monodromy = -1
        else:
            raise ValueError("monodromy must be either 'above' or 'below'")

    def eom(self, field):
        """Compute the field equation of motion due to this source.

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
        field_val = self.charge * 2j * np.pi / field.gridsize**2
        z_index = int(round((self.z - field.zmin) / field.gridsize))
        y0 = self.y * self.monodromy

        laplacian = np.zeros_like(field.field)
        for i in range(field.ny):
            y = (field.ymin + i * field.gridsize) * self.monodromy
            if y >= y0:
                laplacian[:, z_index - 1, i] = field_val
                laplacian[:, z_index, i] = -field_val

        return laplacian * self.monodromy
