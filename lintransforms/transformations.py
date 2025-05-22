"""Library of linear transformations to apply to n-dimensional matrices"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from einops import rearrange

class Transformation(ABC):
    """
    Abstract class for transformations.
    """
    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the transformation to the input data.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the transformation.

        Returns
        -------
        str
            Name of the transformation.
        """
        pass

    def __repr__(self):
        return self.name

class Identity(Transformation):
    """
    Identity transformation of input data.
    """
    def apply(self, x: np.ndarray) -> np.ndarray:
        return x

    @property
    def name(self) -> str:
        return "Identity"

@dataclass
class Rotation(Transformation):
    """
    Rotate the input data by a given angle.
    """
    rotation: np.ndarray = None

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.rotation)

    @property
    def name(self) -> str:
        return f"Rotation"
    
@dataclass
class Inverse(Transformation):
    """
    Inverse the input data.
    """
    def apply(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != x.shape[-2]:
            raise ValueError("Last two dimensions must form square matrices.")
         
        return np.linalg.inv(x)

    @property
    def name(self) -> str:
        return "Inverse"
    
@dataclass
class Reflection(Transformation):
    """
    Reflect the input data.
    """
    axis: int = 0

    def apply(self, x: np.ndarray) -> np.ndarray:
        if not (0 <= self.axis < x.ndim):
            raise ValueError(f"Invalid axis {self.axis} for input shape {x.shape}")
        
        x_copy = np.copy(x)
        dim_names = [f'd{i}' for i in range(x.ndim)]
        x_copy = rearrange(x_copy, f"{' '.join(dim_names)} -> {' '.join(dim_names)}")
        x_copy *= -1
        return x_copy
    
    @property
    def name(self) -> str:
        return f"Reflection(axis={self.axis})"
    
@dataclass
class Dilation(Transformation):
    """
    Dilate the input data by a given factor.
    """
    factor: float = 1.0

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x * self.factor

    @property
    def name(self) -> str:
        return f"Dilation(factor={self.factor})"
    
@dataclass
class Shear(Transformation):
    """
    Shear the input data by a given factor.
    """
    factor: float = 1.0
    axis: int = 0
    source_axis: int = 1

    def apply(self, x: np.ndarray) -> np.ndarray:
        if not (0 <= self.axis < x.ndim and 0 <= self.source_axis < x.ndim):
            raise ValueError(f"Invalid axis pair for shape {x.shape}")
        if self.axis == self.source_axis:
            raise ValueError("axis and source_axis must be different.")
        x_copy = np.copy(x)

        perm = list(range(x.ndim))
        perm.remove(self.axis)
        perm.remove(self.source_axis)
        perm += [self.axis, self.source_axis]
        x_reordered = rearrange(x_copy, f"{' '.join([f'd{i}' for i in range(x.ndim)])} -> {' '.join([f'd{i}' for i in perm])}")

        # Apply shear
        x_reordered[..., 0] += self.factor * x_reordered[..., 1]

        # Restore original axis order
        inverse_perm = np.argsort(perm)
        x_final = rearrange(x_reordered, f"{' '.join([f'd{i}' for i in range(x.ndim)])} -> {' '.join([f'd{i}' for i in inverse_perm])}")
        
        return x_final

    @property
    def name(self) -> str:
        return f"Shear(factor={self.factor}, axis={self.axis})"
    
@dataclass
class Projection(Transformation):
    """
    Project the input data onto a given axis via projection matrix.
    """
    projection: np.ndarray = None

    def apply(self, x: np.ndarray) -> np.ndarray:
            if x.shape[-1] != self.projection.shape[1]:
                raise ValueError("Last dimension must match projection matrix size.")
            return np.matmul(x, self.projection.T)

    @property
    def name(self) -> str:
        return f"Projection(axis={self.axis})"