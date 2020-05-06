import numpy as np


class Field:
    """
    A class used to represent a discretized complex-valued vector field in two
    spatial dimensions.
    """

    def __init__(self, n, zmin, zmax, ymin, ymax, gridsize):
        """Initialize this Field.

        Parameters
        ----------
        n : int
            Number of field components.
        zmin : float
            Leftmost grid point of the domain.
        zmax
            Rightmost grid point of the domain.
        ymin
            Bottom grid point of the domain.
        ymax
            Top grid point of the domain.
        gridsize
            Grid division size.
        """
        self.n = n
        self.zmin = zmin
        self.zmax = zmax
        self.ymin = ymin
        self.ymax = ymax
        self.gridsize = gridsize

        # Number of grid points along each axis
        self.nz = int((zmax - zmin) / gridsize) + 1
        self.ny = int((ymax - ymin) / gridsize) + 1

        # Array of shape (n, nz, ny) representing the field
        self.field = np.zeros((self.n, self.nz, self.ny), dtype=complex)
