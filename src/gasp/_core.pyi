"""Type stubs for the gasp._core Rust extension module."""

from typing import Union
import numpy as np
import numpy.typing as npt

# ============ Size Measures - List-based API ============


def bbox_volume(points: list[list[float]]) -> float:
    """Compute the n-dimensional bounding box hypervolume (list version)."""
    ...


def hull_volume(points: list[list[float]]) -> float:
    """Compute the convex hull volume (list version)."""
    ...


def covariance_volume(points: list[list[float]]) -> float:
    """Compute covariance determinant as n-D ellipsoid volume proxy (list version)."""
    ...


def mean_dispersion(points: list[list[float]]) -> float:
    """Compute mean pairwise Euclidean distance (list version)."""
    ...


# ============ Size Measures - NumPy API ============


def bbox_volume_numpy(points: npt.NDArray[np.float64]) -> float:
    """Compute the n-dimensional bounding box hypervolume (numpy version)."""
    ...


def hull_volume_numpy(points: npt.NDArray[np.float64]) -> float:
    """Compute the convex hull volume (numpy version)."""
    ...


def covariance_volume_numpy(points: npt.NDArray[np.float64]) -> float:
    """Compute covariance determinant as n-D ellipsoid volume proxy (numpy version)."""
    ...


def mean_dispersion_numpy(points: npt.NDArray[np.float64]) -> float:
    """Compute mean pairwise Euclidean distance (numpy version)."""
    ...


# ============ Sampling - List-based API ============


def sample_centroid_uniform(
    points: list[list[float]], count: int, seed: int
) -> list[int]:
    """
    Sample points using centroid + uniform random strategy.

    Args:
        points: List of points (each point is a list of floats)
        count: Number of points to sample
        seed: Random seed for reproducibility

    Returns:
        List of indices of sampled points
    """
    ...


def sample_centroid_fps(points: list[list[float]], count: int, seed: int) -> list[int]:
    """
    Sample points using centroid + farthest point sampling strategy.

    Args:
        points: List of points (each point is a list of floats)
        count: Number of points to sample
        seed: Random seed for reproducibility

    Returns:
        List of indices of sampled points
    """
    ...


def sample_centroid_kmedoids(
    points: list[list[float]], count: int, seed: int, max_iterations: int
) -> list[int]:
    """
    Sample points using centroid + k-medoids strategy.

    Args:
        points: List of points (each point is a list of floats)
        count: Number of points to sample
        seed: Random seed for reproducibility
        max_iterations: Maximum iterations for k-medoids optimization

    Returns:
        List of indices of sampled points
    """
    ...


def sample_centroid_voxel(
    points: list[list[float]], count: int, seed: int, voxel_size: float
) -> list[int]:
    """
    Sample points using centroid + voxel grid strategy.

    Args:
        points: List of points (each point is a list of floats)
        count: Number of points to sample
        seed: Random seed for reproducibility
        voxel_size: Size of voxel grid cells

    Returns:
        List of indices of sampled points
    """
    ...


def sample_centroid_stratified(
    points: list[list[float]],
    count: int,
    seed: int,
    num_strata: int,
    max_iterations: int,
) -> list[int]:
    """
    Sample points using centroid + stratified sampling strategy.

    Args:
        points: List of points (each point is a list of floats)
        count: Number of points to sample
        seed: Random seed for reproducibility
        num_strata: Number of strata (clusters) for stratification
        max_iterations: Maximum iterations for k-means clustering

    Returns:
        List of indices of sampled points
    """
    ...


def sample_uniform(points: list[list[float]], count: int, seed: int) -> list[int]:
    """
    Sample points using pure uniform random strategy (no centroid guarantee).

    Args:
        points: List of points (each point is a list of floats)
        count: Number of points to sample
        seed: Random seed for reproducibility

    Returns:
        List of indices of sampled points
    """
    ...


# ============ Sampling - NumPy API ============


def sample_centroid_uniform_numpy(
    points: npt.NDArray[np.float64], count: int, seed: int
) -> list[int]:
    """
    Sample points using centroid + uniform random strategy (numpy version).

    Args:
        points: NumPy array of shape (n_points, n_dims)
        count: Number of points to sample
        seed: Random seed for reproducibility

    Returns:
        List of indices of sampled points
    """
    ...


def sample_centroid_fps_numpy(
    points: npt.NDArray[np.float64], count: int, seed: int
) -> list[int]:
    """
    Sample points using centroid + farthest point sampling strategy (numpy version).

    Args:
        points: NumPy array of shape (n_points, n_dims)
        count: Number of points to sample
        seed: Random seed for reproducibility

    Returns:
        List of indices of sampled points
    """
    ...


def sample_centroid_kmedoids_numpy(
    points: npt.NDArray[np.float64], count: int, seed: int, max_iterations: int
) -> list[int]:
    """
    Sample points using centroid + k-medoids strategy (numpy version).

    Args:
        points: NumPy array of shape (n_points, n_dims)
        count: Number of points to sample
        seed: Random seed for reproducibility
        max_iterations: Maximum iterations for k-medoids optimization

    Returns:
        List of indices of sampled points
    """
    ...


def sample_centroid_voxel_numpy(
    points: npt.NDArray[np.float64], count: int, seed: int, voxel_size: float
) -> list[int]:
    """
    Sample points using centroid + voxel grid strategy (numpy version).

    Args:
        points: NumPy array of shape (n_points, n_dims)
        count: Number of points to sample
        seed: Random seed for reproducibility
        voxel_size: Size of voxel grid cells

    Returns:
        List of indices of sampled points
    """
    ...


def sample_centroid_stratified_numpy(
    points: npt.NDArray[np.float64],
    count: int,
    seed: int,
    num_strata: int,
    max_iterations: int,
) -> list[int]:
    """
    Sample points using centroid + stratified sampling strategy (numpy version).

    Args:
        points: NumPy array of shape (n_points, n_dims)
        count: Number of points to sample
        seed: Random seed for reproducibility
        num_strata: Number of strata (clusters) for stratification
        max_iterations: Maximum iterations for k-means clustering

    Returns:
        List of indices of sampled points
    """
    ...


def sample_uniform_numpy(
    points: npt.NDArray[np.float64], count: int, seed: int
) -> list[int]:
    """
    Sample points using pure uniform random strategy (numpy version, no centroid guarantee).

    Args:
        points: NumPy array of shape (n_points, n_dims)
        count: Number of points to sample
        seed: Random seed for reproducibility

    Returns:
        List of indices of sampled points
    """
    ...
