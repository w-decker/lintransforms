from .transformations import (
    Transformation,
    Identity,
    Linear,
    Translation,
    Rotation,
    Inverse, 
    Reflection, 
    Dilation, 
    Shear,
    Projection
)

from .solvers import (
    Solver,
    MoorePenrosePseudoInverse,
    LeastSquares,
    OrthogonalProjection,
    Exact,
    MultiTaskLasso,
    Ridge
)

from .utils import (
    correlate,
    issymmetric,
    issquare,
    isidempotent,
    isfullrank
)