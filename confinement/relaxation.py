import numpy as np
_ERASESTR = "                                                                  "


class RelaxationSolver:
    """
    Abstract parent class of RelaxationSolver2D and RelaxationSolver1D.
    """

    def __init__(self, field, func):
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
        """
        self.field = field
        self.func = func

    def update_jacobi(self):
        """Update the field using the Jacobi method of relaxation.

        This method converges slower than the Gauss-Seidel method, but can be
        implemented using vectorized array operations, which may speed up
        the computations.

        Returns
        -------
        error : float
            The current error, defined as max(|f_new - f_old|) / max(|f_new|).
        """
        raise NotImplementedError

    def update_gauss(self):
        """Update the field using the Gauss-Seidel method of relaxation.

        This method converges faster than the Jacobi method, but is implemented
        with explict loops rather than vectorized array operations, which may
        slow down the computations.

        Returns
        -------
        error : float
            The current error, defined as max(|f_new - f_old|) / max(|f_new|).
        """
        raise NotImplementedError

    def solve(self, method='jacobi', tol=1e-4, maxiter=10000, verbose=False):
        """Solve the PDE.

        Parameters
        ----------
        method : str
            Method of solving. Either 'jacobi' for the Jacobi method or 'gauss'
            for the Gauss-Seidel method.
        tol : float
            Error tolerance. The solver will consider the solution to have
            converged once this threshold is reached.
        maxiter : int
            Maximum number of iterations until halting.
        verbose : bool
            If True, print the iteration number and current error after each
            iteration.

        Returns
        -------
        iterations : int
            Number of iterations until the solution converged or maxiter was
            reached.
        error : float
            The error, defined as max(|f_new - f_old|) / max(|f_new|), of the
            final iteration.
        """
        if method == 'jacobi':
            update = self.update_jacobi
        elif method == 'gauss':
            update = self.update_gauss
        else:
            raise ValueError("method must be 'jacobi' or 'gauss'")

        i = 0
        error = np.inf
        for i in range(maxiter):
            error = update()
            if verbose:
                outstr = "Iteration: {}\tError: {:.3g}".format(i + 1, error)
                print("\r" + _ERASESTR + "\r" + outstr, end="\r")
            if error < tol:
                break

        if verbose:
            print()

        return i + 1, error


class RelaxationSolver2D(RelaxationSolver):
    """
    A class used to solve a general discretized 2D second-order PDE of the form
    Dz^2 u + Dy^2 u = f(u, z, y), where u is a complex-valued vector field,
    using the relaxation method.
    """

    def __init__(self, field, func):
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
        """
        super().__init__(field, func)

    def update_jacobi(self):
        """Update the field using the Jacobi method of relaxation.

        This method converges slower than the Gauss-Seidel method, but can be
        implemented using vectorized array operations, which may speed up
        the computations.

        Returns
        -------
        error : float
            The current error, defined as max(|f_new - f_old|) / max(|f_new|).
        """
        # Store the field, grid size, and Laplacian in temporary variables
        f = self.field.field
        h = self.field.gridsize
        laplacian = self.func(self.field)

        # Compute the new values of the field using vectorized operations
        f_new = (f[:, :-2, 1:-1] + f[:, 2:, 1:-1] + f[:, 1:-1, :-2]
                 + f[:, 1:-1, 2:] - h**2 * laplacian[:, 1:-1, 1:-1]) / 4

        # Compute the error
        error = np.max(np.abs(f_new - f[:, 1:-1, 1:-1])) / np.max(np.abs(f_new))

        # Update the field
        f[:, 1:-1, 1:-1] = f_new

        return error

    def update_gauss(self):
        """Update the field using the Gauss-Seidel method of relaxation.

        This method converges faster than the Jacobi method, but is implemented
        with explict loops rather than vectorized array operations, which may
        slow down the computations.

        Returns
        -------
        error : float
            The current error, defined as max(|f_new - f_old|) / max(|f_new|).
        """
        # Store the field, grid size, and Laplacian in temporary variables
        f = self.field.field
        h = self.field.gridsize
        laplacian = self.func(self.field)
        f_old = np.copy(f)

        # Compute the new values of the field using explicit loops
        for i in range(1, self.field.nz - 1):
            for j in range(1, self.field.ny - 1):
                f[:, i, j] = (f[:, i - 1, j] + f[:, i + 1, j] + f[:, i, j - 1]
                              + f[:, i, j + 1] - h**2 * laplacian[:, i, j]) / 4

        # Compute the error
        error = np.max(np.abs(f - f_old)) / np.max(np.abs(f))
        return error


class PoissonSolver2D(RelaxationSolver2D):
    """
    A class used to solve a discretized 2D Poisson problem for a complex-valued
    vector field, using the relaxation method.
    """

    def __init__(self, field, func):
        """Initialize this PoissonSolver2D.

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
            point. It is assumed that func depends only on the y and z grid
            coordinates, and not on the value of the field.
        """
        laplacian = func(field)
        super().__init__(field, lambda f: laplacian)


class RelaxationSolver1D(RelaxationSolver):
    """
    A class used to solve a general discretized 1D second-order PDE of the form
    Dy^2 u = f(u, y), where u is a complex-valued vector field in one variable,
    using the relaxation method.
    """

    def __init__(self, field, func):
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
        """
        super().__init__(field, func)

    def update_jacobi(self):
        """Update the field using the Jacobi method of relaxation.

        This method converges slower than the Gauss-Seidel method, but can be
        implemented using vectorized array operations, which may speed up
        the computations.

        Returns
        -------
        error : float
            The current error, defined as max(|f_new - f_old|) / max(|f_new|).
        """
        # Store the field, grid size, and 2nd derivative in temporary variables
        f = self.field.field
        h = self.field.gridsize
        deriv = self.func(self.field)

        # Compute the new values of the field using vectorized operations
        f_new = (f[:, :-2] + f[:, 2:] - h**2 * deriv[:, 1:-1]) / 2

        # Compute the error
        error = np.max(np.abs(f_new - f[:, 1:-1])) / np.max(np.abs(f_new))

        # Update the field
        f[:, 1:-1] = f_new

        return error

    def update_gauss(self):
        """Update the field using the Gauss-Seidel method of relaxation.

        This method converges faster than the Jacobi method, but is implemented
        with explict loops rather than vectorized array operations, which may
        slow down the computations.

        Returns
        -------
        error : float
            The current error, defined as max(|f_new - f_old|) / max(|f_new|).
        """
        # Store the field, grid size, and 2nd derivative in temporary variables
        f = self.field.field
        h = self.field.gridsize
        deriv = self.func(self.field)
        f_old = np.copy(f)

        # Compute the new values of the field using an explicit loop
        for i in range(1, self.field.ny - 1):
            f[:, i] = (f[:, i - 1] + f[:, i + 1] - h**2 * deriv[:, i]) / 2

        # Compute the error
        error = np.max(np.abs(f - f_old)) / np.max(np.abs(f))
        return error


class PoissonSolver1D(RelaxationSolver1D):
    """
    A class used to solve a discretized 1D Poisson problem for a complex-valued
    vector field, using the relaxation method.
    """

    def __init__(self, field, func):
        """Initialize this PoissonSolver1D.

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
            derivative at each point. It is assumed that func depends only on
            the grid coordinates, and not on the value of the field.
        """
        deriv = func(field)
        super().__init__(field, lambda f: deriv)
