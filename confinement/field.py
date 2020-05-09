import numpy as np


class Field:
    """
    Abstract parent class of Field2D and Field1D.
    """

    def __init__(self, n, gridsize):
        """Initialize this Field.

        Parameters
        ----------
        n : int
            Number of field components.
        gridsize : float
            Grid division size.
        """
        self.n = n
        self.gridsize = gridsize


class Field2D(Field):
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
        zmax : float
            Rightmost grid point of the domain.
        ymin : float
            Lowermost grid point of the domain.
        ymax : float
            Uppermost grid point of the domain.
        gridsize : float
            Grid division size.
        """
        super().__init__(n, gridsize)
        self.zmin = zmin
        self.zmax = zmax
        self.ymin = ymin
        self.ymax = ymax

        # Number of grid points along each axis
        self.nz = int((zmax - zmin) / gridsize) + 1
        self.ny = int((ymax - ymin) / gridsize) + 1

        # Array of shape (n, nz, ny) representing the field
        self.field = np.zeros((self.n, self.nz, self.ny), dtype=complex)


class Field1D(Field):
    """
    A class used to represent a discretized complex-valued vector field in one
    spatial dimension.
    """

    def __init__(self, n, ymin, ymax, gridsize):
        """Initialize this Field1D.

        Parameters
        ----------
        n : int
            Number of field components.
        ymin : float
            Lowermost grid point of the domain.
        ymax : float
            Uppermost grid point of the domain.
        gridsize : float
            Grid division size.
        """
        super().__init__(n, gridsize)
        self.ymin = ymin
        self.ymax = ymax

        # Number of grid points
        self.ny = int((ymax - ymin) / gridsize) + 1

        # Array of shape (n, ny) representing the field
        self.field = np.zeros((self.n, self.ny), dtype=complex)
