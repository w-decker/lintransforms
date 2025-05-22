from .transformations import Transformation
from typing import List

class Pipeline(Transformation):
    """
    Applies a sequence of Transformation objects in order.
    """
    def __init__(self, transforms: List[Transformation]):
        self.transforms = transforms

    def apply(self, x):
        for transform in self.transforms:
            x = transform.apply(x)
        return x

    @property
    def name(self):
        return " -> ".join(t.name for t in self.transforms)