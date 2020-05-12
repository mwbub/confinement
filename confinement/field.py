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
        self.field = None

    def save(self, filename):
        """Save this Field to a file.

        Parameters
        ----------
        filename : str
            The name of the file.

        Returns
        -------
        None
        """
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        """Load a Field from a file.

        Parameters
        ----------
        filename : str
            The name of the file.

        Returns
        -------
        field : Field
            The loaded field.
        """
        raise NotImplementedError

    def gradient(self):
        """Compute the gradient of this Field.

        Returns
        -------
        gradient : ndarray or list of ndarray
            A set of ndarrays corresponding to the derivatives of the field
            along each spatial dimension.
        """
        raise NotImplementedError


class Field2D(Field):
    """
    A class used to represent a discretized complex-valued vector field in two
    spatial dimensions.
    """

    def __init__(self, n, zmin, zmax, ymin, ymax, gridsize):
        """Initialize this Field2D.

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

    def save(self, filename):
        """Save this Field2D to a file.

        Parameters
        ----------
        filename : str
            The name of the file.

        Returns
        -------
        None
        """
        np.savez(filename, field=self.field, n=self.n, zmin=self.zmin,
                 zmax=self.zmax, ymin=self.ymin, ymax=self.ymax,
                 gridsize=self.gridsize)

    @classmethod
    def load(cls, filename):
        """Load a Field2D from a file.

        Parameters
        ----------
        filename : str
            The name of the file.

        Returns
        -------
        field : Field2D
            The loaded field.
        """
        # Load the file and initialize the field
        with np.load(filename) as data:
            field = cls(int(data['n']), float(data['zmin']),
                        float(data['zmax']), float(data['ymin']),
                        float(data['ymax']), float(data['gridsize']))
            field.field = data['field']

        # Check that the field has the correct shape
        shape = (field.n, field.nz, field.ny)
        if field.field.shape != shape:
            raise ValueError("field has shape {}, but expected "
                             "{}".format(field.field.shape, shape))

        return field

    def gradient(self):
        """Compute the gradient of this Field2D.

        Returns
        -------
        gradient : list of ndarray
            A list of two ndarrays of the same shape as self.field,
            corresponding to the derivatives of self.field along the z and y
            axes, respectively.
        """
        return np.gradient(self.field, self.gridsize, axis=(1, 2))


class Field1D(Field):
    """
    A class used to represent a discretized complex-valued vector field in one
    spatial dimension.
    """

    def __init__(self, n, zmin, zmax, gridsize):
        """Initialize this Field1D.

        Parameters
        ----------
        n : int
            Number of field components.
        zmin : float
            Leftmost point of the domain.
        zmax : float
            Rightmost point of the domain.
        gridsize : float
            Grid division size.
        """
        super().__init__(n, gridsize)
        self.zmin = zmin
        self.zmax = zmax

        # Number of grid points
        self.nz = int((zmax - zmin) / gridsize) + 1

        # Array of shape (n, nz) representing the field
        self.field = np.zeros((self.n, self.nz), dtype=complex)

    def save(self, filename):
        """Save this Field1D to a file.

        Parameters
        ----------
        filename : str
            The name of the file.

        Returns
        -------
        None
        """
        np.savez(filename, field=self.field, n=self.n, zmin=self.zmin,
                 zmax=self.zmax, gridsize=self.gridsize)

    @classmethod
    def load(cls, filename):
        """Load a Field1D from a file.

        Parameters
        ----------
        filename : str
            The name of the file.

        Returns
        -------
        field : Field1D
            The loaded field.
        """
        # Load the file and initialize the field
        with np.load(filename) as data:
            field = cls(int(data['n']), float(data['zmin']),
                        float(data['zmax']), float(data['gridsize']))
            field.field = data['field']

        # Check that the field has the correct shape
        shape = (field.n, field.nz)
        if field.field.shape != shape:
            raise ValueError("field has shape {}, but expected "
                             "{}".format(field.field.shape, shape))

        return field

    def gradient(self):
        """Compute the derivative of this Field1D.

        Returns
        -------
        gradient : ndarray
            An ndarray with the same shape as self.field corresponding to the
            derivative at each point.
        """
        return np.gradient(self.field, self.gridsize, axis=1)
