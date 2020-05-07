import numpy as np
from .field import Field


class RelaxationSolver:
    """
    A class used to solve a general discretized 2D second-order PDE of the form
    Dz^2 u + Dy^2 u = f(u, z, y), where u is a complex-valued vector field,
    using the relaxation method.
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
            The current error, defined as the average absolute difference
            between each component of the new field and the old field.
        """
        # Store the field, grid size, and Laplacian in temporary variables
        f = self.field.field
        h = self.field.gridsize
        laplacian = self.func(self.field)

        # Compute the new values of the field using vectorized operations
        f_new = (f[:, :-2, 1:-1] + f[:, 2:, 1:-1] + f[:, 1:-1, :-2]
                 + f[:, 1:-1, 2:] - h**2 * laplacian[:, 1:-1, 1:-1]) / 4

        # Compute the error
        error = np.sum(np.abs(f[:, 1:-1, 1:-1] - f_new)) / f.size

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
            The current error, defined as the average absolute difference
            between each component of the new field and the old field.
        """
        # Store the field, grid size, and Laplacian in temporary variables
        f = self.field.field
        h = self.field.gridsize
        laplacian = self.func(self.field)

        # Compute the new values of the field using explicit loops
        error = 0
        for i in range(1, self.field.nz - 1):
            for j in range(1, self.field.ny - 1):

                # Compute the new value of f[:, i, j]
                f_new = (f[:, i - 1, j] + f[:, i + 1, j] + f[:, i, j - 1]
                         + f[:, i, j + 1] - h**2 * laplacian[:, i, j]) / 4

                # Increment the error
                error += np.sum(np.abs(f[:, i, j] - f_new))

                # Update the field
                f[:, i, j] = f_new

        # Normalize and return the error
        error /= f.size
        return error

    def solve(self, method='jacobi', tol=1e-5, maxiter=1000):
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

        Returns
        -------
        iterations : int
            Number of iterations until the solution converged or maxiter was
            reached.
        """
        if method == 'jacobi':
            update = self.update_jacobi
        elif method == 'gauss':
            update = self.update_gauss
        else:
            raise ValueError("method must be 'jacobi' or 'gauss'")

        i = 0
        for i in range(maxiter):
            error = update()
            if error < tol:
                break

        return i + 1


class PoissonSolver(RelaxationSolver):
    """
    A class used to solve a discretized 2D Poisson problem for a complex-valued
    vector field, using the relaxation method.
    """

    def __init__(self, field, func):
        """Initialize this PoissonSolver.

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
            point. It is assumed that func depends only on the y and z grid
            coordinates, and not on the value of the field.
        """
        laplacian = func(field)
        super().__init__(field, lambda f: laplacian)
