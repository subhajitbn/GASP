"""Type stubs for the gasp._core Rust extension module."""

from typing import Union
import numpy as np
import numpy.typing as npt

# List-based API
def bbox_volume(points: list[list[float]]) -> float:
    """Compute the n-dimensional bounding box hypervolume (list version)."""
    ...

def covariance_volume(points: list[list[float]]) -> float:
    """Compute covariance determinant as n-D ellipsoid volume proxy (list version)."""
    ...

def mean_dispersion(points: list[list[float]]) -> float:
    """Compute mean pairwise Euclidean distance (list version)."""
    ...

# NumPy API
def bbox_volume_numpy(points: npt.NDArray[np.float64]) -> float:
    """Compute the n-dimensional bounding box hypervolume (numpy version)."""
    ...

def covariance_volume_numpy(points: npt.NDArray[np.float64]) -> float:
    """Compute covariance determinant as n-D ellipsoid volume proxy (numpy version)."""
    ...

def mean_dispersion_numpy(points: npt.NDArray[np.float64]) -> float:
    """Compute mean pairwise Euclidean distance (numpy version)."""
    ...
