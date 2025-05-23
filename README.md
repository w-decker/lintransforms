# Lintransforms

Apply various linear transformations on $n$ -dimensional matrices and solve for their transformation matrix. 

# Installation

```shell
pip install git+https://github.com/w-decker/lintransforms.git
```

## Transformations

There are nine transformations to impose: `Identity`, `Linear`, `Translation`, `Rotation`, `Inverse`, `Reflection`, `Dilation`, `Shear` and `Projection`.

## Solvers
There are five general solvers: `ExactSolver`, `MoorePenrosePseudoInverse` and `LeastSquares`, `MultiTaskLasso` and `Ridge` as well as a projection-specific solver: `SolveOrthogonalProjection`. 

# Example usage

```python
from lintransforms.transformations import Rotation
from lintransforms.solvers import LeastSquares
import numpy as np

# random matrix
X = np.random.randn(5, 5)

# rotation matrix
Q, _ = np.linalg.qr(np.random.randn(5, 5))  # Orthonormal matrix
rotation = Rotation(Q)

# apply transformation
Y = rotation.apply(X)

# recover transformation matrix
solver = LeastSquares()
W = solver.solve(X, Y)
```

You can also apply multiple transformations and test out multiple solvers sequentially with `lintransforms.pipeline.Pipeline`. 

```python
from lintransforms.pipeline import Pipeline
from lintransforms.transformations import Rotation, Identity
from lintransforms.solvers import Exact, LeastSquares
import numpy as np

# random matrix
X = np.random.randn(5, 5)
Q, _ = np.linalg.qr(np.random.randn(5, 5))  # Orthonormal matrix

# pipeline
pipeline = Pipeline([
    Rotation(Q),
    Identity()
],
[
    Exact(),
    LeastSquares()
])

# apply transformations
Y = pipeline.apply(X)

# recover transformation matrices for different solvers
solutions_dict = pipeline.solve(X, Y)
```