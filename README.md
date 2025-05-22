# TaskTransforms-simulations

Apply various linear transformations on $n$ -dimensional matrices and solve for their transformation matrix. 

## Transformations

There are seven transformations to impose: `Identity`, `Rotation`, `Inverse`, `Reflection`, `Dilation`, `Shear` and `Projection`.

## Solvers
There are two general solvers: `MoorePenrosePseudoInverse` and `LeastSquares` as well as a projection-specific `SolveOrthogonalProjection`. 

# Example usage

```python
from src.transformations import Rotation
from src.solvers import LeastSquares

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
