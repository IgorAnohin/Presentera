from dataclasses import dataclass


@dataclass
class VisiblePoint2D:
    x: float
    y: float
    visibility: float = 1.0