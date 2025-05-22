from abc import ABC, abstractmethod
import numpy as np

class BaseSolver(ABC):

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

class MoorePenrosePseudoInverse(BaseSolver):
    """
    Moore-Penrose pseudo-inverse solver.
    """
    def solve(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(x) @ y

    @property
    def name(self) -> str:
        return "Moore-Penrose Pseudo-Inverse"
    
class LeastSquares(BaseSolver):
    """
    Least squares solver.
    """
    def solve(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(x, y, rcond=None)[0]
    
    @property
    def name(self) -> str:
        return "Least Squares"
    
class SolveOrthogonalProjection(BaseSolver):
    """
    Projection solver.
    """

    def solve(self, x: np.ndarray) -> np.ndarray:
        A = x
        P = A @ np.linalg.pinv(A) # for rank deficient A
        return P
    
    @property
    def name(self) -> str:
        return "Orthogonal Projection"