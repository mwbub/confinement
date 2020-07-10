import numpy as np
from scipy.integrate import simps


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

    def energy_density(self):
        """Compute the energy density of this field due to its gradient.

        Returns
        -------
        energy_density : ndarray
            The energy density at each point.
        """
        raise NotImplementedError

    def energy(self):
        """Compute the energy of this Field due to its gradient.

        Returns
        -------
        energy : float
            The total energy.
        """
        raise NotImplementedError


class Field2D(Field):
    """
    A class used to represent a discretized complex-valued vector field in two
    spatial dimensions.

    Attributes
    ----------
    n : int
        Number of field components.
    nz : int
        Number of grid points along the z-axis.
    ny : int
        Number of grid points along the y-axis.
    field : ndarray
        Value of the field at each point. Has shape (n, nz, ny).
    z : ndarray
        Array of grid points along the z-axis.
    y : ndarray
        Array of grid points along the y-axis.
    zmin : float
        Lower bound of the z-axis.
    zmax : float
        Upper bound of the z-axis.
    ymin : float
        Lower bound of the y-axis.
    ymax : float
        Upper bound of the y-axis.
    gridsize : float
        Grid division size.

    Notes
    -----
    Depending on the grid size, the attributes `zmax` and `ymax` are not
    necessarily equivalent to the largest values in the `z` and `y` arrays,
    respectively. Instead, they are merely upper bounds.
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

        # Arrays of grid points for each axis
        self.z = np.linspace(zmin, zmin + gridsize * (self.nz - 1), self.nz)
        self.y = np.linspace(ymin, ymin + gridsize * (self.ny - 1), self.ny)

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

    def gradient(self, z_jumps=None, y_jumps=None):
        """Compute the gradient of this Field2D.

        Parameters
        ----------
        z_jumps : list of float
            List of z coordinates of discontinuities parallel to the y-axis.
            The one-sided derivative will be taken at these points.
        y_jumps : list of float
            List of y coordinates of discontinuities parallel to the z-axis.
            The one-sided derivative will be taken at these points.

        Returns
        -------
        gradient : list of ndarray
            A list of two ndarrays of the same shape as self.field,
            corresponding to the derivatives of self.field along the z and y
            axes, respectively.
        """
        if z_jumps is None:
            z_jumps = []
        if y_jumps is None:
            y_jumps = []

        gradient = np.gradient(self.field, self.gridsize, axis=(1, 2))

        # Compute the one-sided derivative for discontinuities
        f = self.field
        h = self.gridsize
        for z in z_jumps:
            i = int(round((z - self.zmin) / self.gridsize))
            gradient[0][:, i - 1, :] = (f[:, i - 1, :] - f[:, i - 2, :]) / h
            gradient[0][:, i, :] = (f[:, i + 1, :] - f[:, i, :]) / h
        for y in y_jumps:
            i = int(round((y - self.ymin) / self.gridsize))
            gradient[1][:, :, i - 1] = (f[:, :, i - 1] - f[:, :, i - 2]) / h
            gradient[1][:, :, i] = (f[:, :, i + 1] - f[:, :, i]) / h

        return gradient

    def energy_density(self, K=None, z_jumps=None, y_jumps=None):
        """Compute the energy density of this Field2D due to its gradient.

        Parameters
        ----------
        K : ndarray
            Array of shape (N-1, N-1) giving the inverse of the Kahler metric.
            If not provided, then this defaults to the identity.
        z_jumps : list of float
            List of z coordinates of discontinuities parallel to the y-axis.
            The one-sided derivative will be taken at these points.
        y_jumps : list of float
            List of y coordinates of discontinuities parallel to the z-axis.
            The one-sided derivative will be taken at these points.

        Returns
        -------
        energy_density : ndarray
            The energy density at each point. Has shape (nz, ny).
        """
        dfdz, dfdy = self.gradient(z_jumps=z_jumps, y_jumps=y_jumps)
        if K is None:
            return np.sum(np.abs(dfdz)**2 + np.abs(dfdy)**2, axis=0)
        else:
            K_inv = np.linalg.inv(K)
            sum1 = np.abs(np.einsum('i...,ij,j...', dfdz, K_inv, np.conj(dfdz)))
            sum2 = np.abs(np.einsum('i...,ij,j...', dfdy, K_inv, np.conj(dfdy)))
            return sum1 + sum2

    def energy(self, K=None, z_jumps=None, y_jumps=None):
        """Compute the energy of this Field2D due to its gradient.

        Parameters
        ----------
        K : ndarray
            Array of shape (N-1, N-1) giving the inverse of the Kahler metric.
            If not provided, then this defaults to the identity.
        z_jumps : list of float
            List of z coordinates of discontinuities parallel to the y-axis.
            The one-sided derivative will be taken at these points.
        y_jumps : list of float
            List of y coordinates of discontinuities parallel to the z-axis.
            The one-sided derivative will be taken at these points.

        Returns
        -------
        energy : float
            The total energy.
        """
        # Compute the energy density and repeatedly integrate over all axes
        density = self.energy_density(K=K, z_jumps=z_jumps, y_jumps=y_jumps)
        return simps(simps(density, x=self.y), x=self.z)


class Field1D(Field):
    """
    A class used to represent a discretized complex-valued vector field in one
    spatial dimension.

    Attributes
    ----------
    n : int
        Number of field components.
    nz : int
        Number of grid points along the z-axis.
    field : ndarray
        Value of the field at each point. Has shape (n, nz).
    z : ndarray
        Array of grid points along the z-axis.
    zmin : float
        Lower bound of the z-axis.
    zmax : float
        Upper bound of the z-axis.
    gridsize : float
        Grid division size.

    Notes
    -----
    Depending on the grid size, the attribute `zmax` is not necessarily
    equivalent to the largest value in the `z` array. Instead, it is merely an
    upper bound.
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

        # Number of grid points and array of points
        self.nz = int((zmax - zmin) / gridsize) + 1
        self.z = np.linspace(zmin, zmin + gridsize * (self.nz - 1), self.nz)

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

    def gradient(self, z_jumps=None):
        """Compute the derivative of this Field1D.

        Parameters
        ----------
        z_jumps : list of float
            List of z coordinates of discontinuities. The one-sided derivative
            will be taken at these points.

        Returns
        -------
        gradient : ndarray
            An ndarray with the same shape as self.field corresponding to the
            derivative at each point.
        """
        if z_jumps is None:
            z_jumps = []

        gradient = np.gradient(self.field, self.gridsize, axis=1)

        # Compute the one-sided derivative for discontinuities
        f = self.field
        h = self.gridsize
        for z in z_jumps:
            i = int(round((z - self.zmin) / self.gridsize))
            gradient[:, i - 1] = (f[:, i - 1] - f[:, i - 2]) / h
            gradient[:, i] = (f[:, i + 1] - f[:, i]) / h

        return gradient

    def energy_density(self, K=None, z_jumps=None):
        """Compute the energy density of this Field1D due to its gradient.

        Parameters
        ----------
        K : ndarray
            Array of shape (N-1, N-1) giving the inverse of the Kahler metric.
            If not provided, then this defaults to the identity.
        z_jumps : list of float
            List of z coordinates of discontinuities. The one-sided derivative
            will be taken at these points.

        Returns
        -------
        energy_density : ndarray
            The energy density at each point. Has shape (nz,).
        """
        dfdz = self.gradient(z_jumps=z_jumps)
        if K is None:
            return np.sum(np.abs(dfdz)**2, axis=0)
        else:
            K_inv = np.linalg.inv(K)
            return np.abs(np.einsum('i...,ij,j...', dfdz, K_inv, np.conj(dfdz)))

    def energy(self, K=None, z_jumps=None):
        """Compute the energy of this Field1D due to its gradient.

        Parameters
        ----------
        K : ndarray
            Array of shape (N-1, N-1) giving the inverse of the Kahler metric.
            If not provided, then this defaults to the identity.
        z_jumps : list of float
            List of z coordinates of discontinuities. The one-sided derivative
            will be taken at these points.

        Returns
        -------
        energy : float
            The total energy.
        """
        # Compute the energy density and integrate
        density = self.energy_density(K=K, z_jumps=z_jumps)
        return simps(density, x=self.z)
