from .transformations import Transformation
from typing import List
from dataclasses import dataclass

@dataclass
class Pipeline(Transformation):
    """
    Applies a sequence of Transformation objects in order.
    """
    transforms: List[Transformation]

    def apply(self, x):
        for transform in self.transforms:
            x = transform.apply(x)
        return x

    @property
    def name(self):
        return " -> ".join(t.name for t in self.transforms)