from abc import ABC, abstractmethod
import numpy as np
from warnings import warn
from dataclasses import dataclass

from .utils import isfullrank, issquare

class Solver(ABC):

    @abstractmethod
    def solve(self, x: np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        Approximate the transformation of the input data to the output data.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        y : np.ndarray
            Output data.

        Returns
        -------
        np.ndarray
            Solution
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the solver.

        Returns
        -------
        str
            Name of the solver.
        """
        pass

    def __repr__(self):
        return self.name
    
class ExactSolver(Solver):
    """
    Exact solver.
    """
    def solve(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not issquare(x):
            raise np.linalg.LinAlgError("Matrix is not square.")
        
        if not isfullrank(x):
            raise np.linalg.LinAlgError("Matrix is rank deficient.")
        
        return np.linalg.solve(x, y)

    @property
    def name(self) -> str:
        return "Exact Solver"

class MoorePenrosePseudoInverse(Solver):
    """
    Moore-Penrose pseudo-inverse solver.
    """
    def solve(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(x) @ y

    @property
    def name(self) -> str:
        return "Moore-Penrose Pseudo-Inverse"
    
class LeastSquares(Solver):
    """
    Least squares solver.
    """
    def solve(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(x, y, rcond=None)[0]
    
    @property
    def name(self) -> str:
        return "Least Squares"
    
class SolveOrthogonalProjection(Solver):
    """
    Projection solver.
    """

    def solve(self, x: np.ndarray) -> np.ndarray:
        if isfullrank(x):
            A = x
            AtA = A.T @ A
            inv_AtA = np.linalg.inv(AtA)
            P = A @ inv_AtA @ A.T
            return P
        else:
            warn("Matrix is rank deficient. Using pseudo-inverse instead.")
            A = x
            P = A @ np.linalg.pinv(A) # for rank deficient A
            return P
    
    @property
    def name(self) -> str:
        return "Orthogonal Projection"

class MultiTaskLasso(Solver):
    """
    LASSO solver.
    Wrapper around sklearn.linear_model.MultiTaskLasso.
    """
    def __init__(self, alpha=1.0, **kwargs):
        self.alpha = alpha
        self.kwargs = kwargs
 
    def solve(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.linear_model import MultiTaskLasso as SklearnMultiTaskLasso
        model = SklearnMultiTaskLasso(alpha=self.alpha, **self.kwargs)
        model.fit(x, y)
        return model.coef_

    @property
    def name(self) -> str:
        return "LASSO"

class Ridge(Solver):
    """
    Ridge regression solver.
    Wrapper around sklearn.linear_model.Ridge.
    """
    def __init__(self, alpha=1.0, **kwargs):
        self.alpha = alpha
        self.kwargs = kwargs

    def solve(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.linear_model import Ridge as SklearnRidge
        model = SklearnRidge(alpha=self.alpha, **self.kwargs)
        model.fit(x, y)
        return model.coef_

    @property
    def name(self) -> str:
        return "Ridge"