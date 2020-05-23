import numpy as np
from numba import jit
from .field import Field2D
_ERASESTR = "                                                                  "


class RelaxationSolver:
    """
    Abstract parent class of RelaxationSolver2D and RelaxationSolver1D.
    """

    def __init__(self, field, func, constant=None):
        """Initialize this RelaxationSolver.

        Parameters
        ----------
        field : Field
            The vector field which defines the grid and where the solution will
            ultimately be stored. The solver assumes that the boundary
            conditions for the field have already been set.
        func : callable(Field)
            The function which defines the Laplacian of the field. This should
            take as its argument the field, and return a complex-valued array
            of the same shape as field.field which gives the Laplacian at each
            point.
        constant : array_like
            A constant term to add to the Laplacian. This should be an array
            with the same shape as field.field.
        """
        self.field = field

        if func is not None:
            self.func = func
        else:
            zeros = np.zeros_like(field.field)
            self.func = lambda f: zeros

        if constant is not None:
            self.constant = np.array(constant)
        else:
            self.constant = 0

    def solve(self, method='gauss', tol=1e-9, maxiter=10000, omega=1.,
              verbose=False):
        """Solve the PDE.

        Parameters
        ----------
        method : str
            Method of solving. Either 'jacobi' for the Jacobi method or 'gauss'
            for the Gauss-Seidel method.
        tol : float
            Relative error tolerance. The solver will consider the solution to
            have converged once this threshold is reached.
        maxiter : int
            Maximum number of iterations until halting.
        omega : float
            The relaxation factor, used for solving with successive
            over-relaxation or under-relaxation.
        verbose : bool
            If True, print the iteration number and current error after each
            iteration.

        Returns
        -------
        iterations : int
            Number of iterations until the solution converged or maxiter was
            reached.
        error : float
            The relative error, defined as max(|f_new - f_old|) / max(|f_new|),
            of the final iteration.
        """
        # Set the update method
        if method == 'jacobi':
            update = self._update_jacobi
        elif method == 'gauss':
            update = self._update_gauss
        else:
            raise ValueError("method must be 'jacobi' or 'gauss'")

        # Update until the error is small or the max iteration count is reached
        i = 0
        error = np.inf
        for i in range(maxiter):
            error = update(omega)
            if verbose:
                outstr = "Iteration: {}\tError: {:.3g}".format(i + 1, error)
                print("\r" + _ERASESTR + "\r" + outstr, end="\r")
            if error < tol:
                break

        # Print a newline before returning if using verbose mode
        if verbose:
            print()

        return i + 1, error

    def _update_jacobi(self, omega):
        """Update the field using the Jacobi method of relaxation.

        This method converges slower than the Gauss-Seidel method, but can be
        implemented using vectorized array operations, which may speed up
        the computations.

        Parameters
        ----------
        omega : float
            The relaxation factor, used for solving with successive
            over-relaxation or under-relaxation.

        Returns
        -------
        error : float
            The current error, defined as max(|f_new - f_old|) / max(|f_new|).
        """
        raise NotImplementedError

    def _update_gauss(self, omega):
        """Update the field using the Gauss-Seidel method of relaxation.

        This method converges faster than the Jacobi method, but is implemented
        with explict loops rather than vectorized array operations, which may
        slow down the computations. The speed of the loops are enhanced using
        the numba just-in-time compiler.

        Parameters
        ----------
        omega : float
            The relaxation factor, used for solving with successive
            over-relaxation or under-relaxation.

        Returns
        -------
        error : float
            The current error, defined as max(|f_new - f_old|) / max(|f_new|).
        """
        raise NotImplementedError


class RelaxationSolver2D(RelaxationSolver):
    """
    A class used to solve a general discretized 2D second-order PDE of the form
    Dz^2 u + Dy^2 u = f(u, z, y), where u is a complex-valued vector field,
    using the relaxation method.
    """

    def __init__(self, field, func, constant=None):
        """Initialize this RelaxationSolver2D.

        Parameters
        ----------
        field : Field2D
            The vector field which defines the grid and where the solution will
            ultimately be stored. The solver assumes that the boundary
            conditions for the field have already been set.
        func : callable(Field2D)
            The function which defines the Laplacian of the field. This should
            take as its argument the field, and return a complex-valued array
            of the same shape as field.field which gives the Laplacian at each
            point.
        constant : array_like
            A constant term to add to the Laplacian. This should be an array
            with the same shape as field.field.
        """
        super().__init__(field, func, constant=constant)

    def symmetric_solve(self, tol=1e-9, maxiter=10000, omega=1., verbose=False):
        """Solve the PDE assuming mirror symmetry about the centre of the grid.

        This method solves the PDE on the left half of the field using the
        Gauss-Seidel method, while imposing Neumann boundary conditions on the
        centre column of the grid. The final field is then found by reflecting
        the left half of the field onto the right half.

        Parameters
        ----------
        tol : float
            Relative error tolerance. The solver will consider the solution to
            have converged once this threshold is reached.
        maxiter : int
            Maximum number of iterations until halting.
        omega : float
            The relaxation factor, used for solving with successive
            over-relaxation or under-relaxation.
        verbose : bool
            If True, print the iteration number and current error after each
            iteration.

        Returns
        -------
        iterations : int
            Number of iterations until the solution converged or maxiter was
            reached.
        error : float
            The relative error, defined as max(|f_new - f_old|) / max(|f_new|),
            of the final iteration.
        """
        # Create a new field to contain the left half of self.field
        centre_index = (self.field.nz - 1) // 2
        centre = self.field.z[centre_index]
        left_field = Field2D(self.field.n, self.field.zmin, centre,
                             self.field.ymin, self.field.ymax,
                             self.field.gridsize)
        left_field.field[:] = self.field.field[:, :centre_index + 1]

        # Update until the error is small or the max iteration count is reached
        i = 0
        error = np.inf
        for i in range(maxiter):
            error = self._symmetric_gauss(left_field, omega)
            if verbose:
                outstr = "Iteration: {}\tError: {:.3g}".format(i + 1, error)
                print("\r" + _ERASESTR + "\r" + outstr, end="\r")
            if error < tol:
                break

        # Print a newline before returning if using verbose mode
        if verbose:
            print()

        # Copy and reflect the left half of the field into the original field
        self.field.field[:, :centre_index + 1] = left_field.field
        if self.field.nz % 2 == 0:
            self.field.field[:, centre_index + 1:] = left_field.field[:, ::-1]
        else:
            self.field.field[:, centre_index + 1:] = left_field.field[:, -2::-1]

        return i + 1, error

    def _update_jacobi(self, omega):
        """Update the field using the Jacobi method of relaxation.

        This method converges slower than the Gauss-Seidel method, but can be
        implemented using vectorized array operations, which may speed up
        the computations.

        Parameters
        ----------
        omega : float
            The relaxation factor, used for solving with successive
            over-relaxation or under-relaxation.

        Returns
        -------
        error : float
            The current error, defined as max(|f_new - f_old|) / max(|f_new|).
        """
        # Store the field, grid size, and Laplacian in temporary variables
        f = self.field.field
        h = self.field.gridsize
        laplacian = self.func(self.field) + self.constant

        # Compute the new values of the field using vectorized operations
        residual = (f[:, :-2, 1:-1] + f[:, 2:, 1:-1] + f[:, 1:-1, :-2]
                    + f[:, 1:-1, 2:] - 4 * f[:, 1:-1, 1:-1]
                    - h**2 * laplacian[:, 1:-1, 1:-1])
        delta = residual * omega / 4

        # Update the field
        f[:, 1:-1, 1:-1] += delta

        # Compute the error
        error = np.max(np.abs(delta)) / np.max(np.abs(f))
        return error

    def _update_gauss(self, omega):
        """Update the field using the Gauss-Seidel method of relaxation.

        This method converges faster than the Jacobi method, but is implemented
        with explict loops rather than vectorized array operations, which may
        slow down the computations. The speed of the loops are enhanced using
        the numba just-in-time compiler.

        Parameters
        ----------
        omega : float
            The relaxation factor, used for solving with successive
            over-relaxation or under-relaxation.

        Returns
        -------
        error : float
            The current error, defined as max(|f_new - f_old|) / max(|f_new|).
        """
        # Copy the field to compute the error later
        f_old = np.copy(self.field.field)

        # Update the field using compiled code
        laplacian = self.func(self.field) + self.constant
        _update_gauss2d(self.field.field, self.field.gridsize, laplacian, omega)

        # Compute the error
        delta = self.field.field - f_old
        error = np.max(np.abs(delta)) / np.max(np.abs(self.field.field))
        return error

    def _symmetric_gauss(self, field, omega):
        """Update a mirror symmetric field using Gauss-Seidel relaxation.

        Parameters
        ----------
        field : Field2D
            The left half of the field to update. Here, field.field should be
            equivalent to self.field.field[:, :(self.field.nz - 1) // 2 + 1].
        omega : float
            The relaxation factor, used for solving with successive
            over-relaxation or under-relaxation.

        Returns
        -------
        error : float
            The current error, defined as max(|f_new - f_old|) / max(|f_new|).
        """
        # Copy the field to compute the error later
        f_old = np.copy(field.field)

        # Update the field using compiled code
        laplacian = self.func(field) + self.constant[:, :field.nz]
        _symmetric_gauss2d(field.field, field.gridsize, laplacian, omega)

        # Compute the error
        delta = field.field - f_old
        error = np.max(np.abs(delta)) / np.max(np.abs(field.field))
        return error


class PoissonSolver2D(RelaxationSolver2D):
    """
    A class used to solve a discretized 2D Poisson problem for a complex-valued
    vector field, using the relaxation method.
    """

    def __init__(self, field, laplacian):
        """Initialize this PoissonSolver2D.

        Parameters
        ----------
        field : Field2D
            The vector field which defines the grid and where the solution will
            ultimately be stored. The solver assumes that the boundary
            conditions for the field have already been set.
        laplacian : array_like
            The array which defines the Laplacian of the field at each point.
            This must have the same shape as field.field.
        """
        super().__init__(field, None, constant=laplacian)


class RelaxationSolver1D(RelaxationSolver):
    """
    A class used to solve a general discretized 1D second-order PDE of the form
    Dy^2 u = f(u, y), where u is a complex-valued vector field in one variable,
    using the relaxation method.
    """

    def __init__(self, field, func, constant=None):
        """Initialize this RelaxationSolver1D.

        Parameters
        ----------
        field : Field1D
            The vector field which defines the grid and where the solution will
            ultimately be stored. The solver assumes that the boundary
            conditions for the field have already been set.
        func : callable(Field1D)
            The function which defines the second derivative of the field. This
            should take as its argument the field, and return a complex-valued
            array of the same shape as field.field which gives the second
            derivative at each point.
        constant : array_like
            A constant term to add to the second derivative. This should be an
            array with the same shape as field.field.
        """
        super().__init__(field, func, constant=constant)

    def _update_jacobi(self, omega):
        """Update the field using the Jacobi method of relaxation.

        This method converges slower than the Gauss-Seidel method, but can be
        implemented using vectorized array operations, which may speed up
        the computations.

        Parameters
        ----------
        omega : float
            The relaxation factor, used for solving with successive
            over-relaxation or under-relaxation.

        Returns
        -------
        error : float
            The current error, defined as max(|f_new - f_old|) / max(|f_new|).
        """
        # Store the field, grid size, and 2nd derivative in temporary variables
        f = self.field.field
        h = self.field.gridsize
        deriv = self.func(self.field) + self.constant

        # Compute the new values of the field using vectorized operations
        residual = f[:, :-2] + f[:, 2:] - h**2 * deriv[:, 1:-1] - 2 * f[:, 1:-1]
        delta = residual * omega / 2

        # Update the field
        f[:, 1:-1] += delta

        # Compute the error
        error = np.max(np.abs(delta)) / np.max(np.abs(f))
        return error

    def _update_gauss(self, omega):
        """Update the field using the Gauss-Seidel method of relaxation.

        This method converges faster than the Jacobi method, but is implemented
        with explict loops rather than vectorized array operations, which may
        slow down the computations. The speed of the loops are enhanced using
        the numba just-in-time compiler.

        Parameters
        ----------
        omega : float
            The relaxation factor, used for solving with successive
            over-relaxation or under-relaxation.

        Returns
        -------
        error : float
            The current error, defined as max(|f_new - f_old|) / max(|f_new|).
        """
        # Copy the field to compute the error later
        f_old = np.copy(self.field.field)

        # Update the field using compiled code
        deriv = self.func(self.field) + self.constant
        _update_gauss1d(self.field.field, self.field.gridsize, deriv, omega)

        # Compute the error
        delta = self.field.field - f_old
        error = np.max(np.abs(delta)) / np.max(np.abs(self.field.field))
        return error


class PoissonSolver1D(RelaxationSolver1D):
    """
    A class used to solve a discretized 1D Poisson problem for a complex-valued
    vector field, using the relaxation method.
    """

    def __init__(self, field, deriv):
        """Initialize this PoissonSolver1D.

        Parameters
        ----------
        field : Field1D
            The vector field which defines the grid and where the solution will
            ultimately be stored. The solver assumes that the boundary
            conditions for the field have already been set.
        deriv : array_like
            The array which defines the second derivative of the field at each
            point. This must have the same shape as field.field.
        """
        super().__init__(field, None, constant=deriv)


@jit(nopython=True)
def _update_gauss2d(f, h, laplacian, omega):
    """Update a 2D field using the Gauss-Seidel method.

    Parameters
    ----------
    f : ndarray
        The field. Has shape (n, nz, ny), where n is the number of components.
    h : float
        The gridsize.
    laplacian : ndarray
        Array of the laplacian at each point. Has the same shape as f.
    omega : float
        The relaxation parameter.

    Returns
    -------
    None
    """
    for i in range(1, f.shape[1] - 1):
        for j in range(1, f.shape[2] - 1):
            residual = (f[:, i - 1, j] + f[:, i + 1, j] + f[:, i, j - 1]
                        + f[:, i, j + 1] - 4 * f[:, i, j]
                        - h**2 * laplacian[:, i, j])
            delta = residual * omega / 4
            f[:, i, j] += delta


@jit(nopython=True)
def _update_gauss1d(f, h, deriv, omega):
    """Update a 1D field using the Gauss-Seidel method.

    Parameters
    ----------
    f : ndarray
        The field. Has shape (n, nz), where n is the number of components.
    h : float
        The gridsize.
    deriv : ndarray
        Array of the second derivative at each point. Has the same shape as f.
    omega : float
        The relaxation parameter.

    Returns
    -------
    None
    """
    for i in range(1, f.shape[1] - 1):
        residual = f[:, i - 1] + f[:, i + 1] - 2 * f[:, i] - h**2 * deriv[:, i]
        delta = residual * omega / 2
        f[:, i] += delta


@jit(nopython=True)
def _symmetric_gauss2d(f, h, laplacian, omega):
    """Update a mirror-symmetric 2D field using Gauss-Seidel relaxation.

    Parameters
    ----------
    f : ndarray
        The left half of the field. Has shape (n, (nz - 1) // 2 + 1, ny), where
        n is the number of components.
    h : float
        The gridsize.
    laplacian : ndarray
        Array of the laplacian at each point. Has the same shape as f.
    omega : float
        The relaxation parameter.

    Returns
    -------
    None
    """
    # Update the interior points of the field
    _update_gauss2d(f, h, laplacian, omega)

    # Update the right boundary assuming mirror symmetry across the boundary
    if f.shape[1] % 2 == 0:
        for j in range(1, f.shape[2] - 1):
            residual = (f[:, -2, j] + f[:, -1, j - 1] + f[:, -1, j + 1]
                        - 3 * f[:, -1, j] - h**2 * laplacian[:, -1, j])
            delta = residual * omega / 4
            f[:, -1, j] += delta
    else:
        for j in range(1, f.shape[2] - 1):
            residual = (2 * f[:, -2, j] + f[:, -1, j - 1] + f[:, -1, j + 1]
                        - 4 * f[:, -1, j] - h**2 * laplacian[:, -1, j])
            delta = residual * omega / 4
            f[:, -1, j] += delta
