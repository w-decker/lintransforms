from typing import List, final
from dataclasses import dataclass, field
import numpy as np

from .transformations import Transformation
from .solvers import Solver

@final
@dataclass(slots=True)
class Pipeline:
    """
    Applies a sequence of Transformation objects and/or Solver objects in order.
    """
    
    transformations: List[Transformation] = field(default_factory=list)
    solvers: List[Solver] = field(default_factory=list)

    def apply(self, x):
        for transform in self.transformations:
            x = transform.apply(x)
        return x
    
    def solve(self, x, y):
        solutions = {}
        for solver in self.solvers:
            try:
                solutions[solver.name] = solver.solve(x, y)
            except np.linalg.LinAlgError as e:
                solutions[solver.name] = str(e)
        return solutions