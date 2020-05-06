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
            The function which defines the Laplacain of the field. This should
            take as its argument the field, and return a complex-valued array
            of the same shape as field.field which gives the Laplacian at each
            point.
        """
        self.field = field
        self.func = func


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
