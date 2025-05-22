# TaskTransforms-simulations

Apply various linear transformations on $n$ -dimensional matrices and solve for their transformation matrix. 

# Installation

```shell
pip install git+https://github.com/w-decker/lintransforms.git
```

## Transformations

There are seven transformations to impose: `Identity`, `Rotation`, `Inverse`, `Reflection`, `Dilation`, `Shear` and `Projection`.

## Solvers
There are two general solvers: `MoorePenrosePseudoInverse` and `LeastSquares` as well as a projection-specific `SolveOrthogonalProjection`. 

# Example usage

```python
from lintransforms.transformations import Rotation
from lintransforms.solvers import LeastSquares
import numpy as np

# random matrix
X = np.random.randn(100, 5)

# rotation matrix
Q, _ = np.linalg.qr(np.random.randn(5, 5))  # Orthonormal matrix
rotation = Rotation(Q)

# apply transformation
Y = rotation.apply(X)

# recover transformation matrix
solver = LeastSquares()
W = solver.solve(X, Y)
```

You can also apply multiple transformations sequentially with `lintransforms.pipeline.Pipeline`. 

```python
from lintransforms.pipeline import Pipeline
from lintransforms.transformations import Rotation, Identity
from lintransforms.solvers import LeastSquares
import numpy as np

# random matrix
X = np.random.randn(100, 5)
Q, _ = np.linalg.qr(np.random.randn(5, 5))  # Orthonormal matrix

# pipeline
pipeline = Pipeline([
    Rotation(Q),
    Identity(),
])

# apply transformations
Y = pipeline.apply(X)
```