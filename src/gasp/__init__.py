"""
GASP: Geometric Analysis and Sampling Package
A high-performance library for point cloud analysis and sampling.
"""

from gasp._core import (
    # Size Measures - List API
    bbox_volume as _bbox_volume_list,
    hull_volume as _hull_volume_list,
    covariance_volume as _covariance_volume_list,
    mean_dispersion as _mean_dispersion_list,
    # Size Measures - NumPy API
    bbox_volume_numpy as _bbox_volume_numpy,
    hull_volume_numpy as _hull_volume_numpy,
    covariance_volume_numpy as _covariance_volume_numpy,
    mean_dispersion_numpy as _mean_dispersion_numpy,
    # Sampling - List API
    sample_centroid_uniform as _sample_centroid_uniform_list,
    sample_centroid_fps as _sample_centroid_fps_list,
    sample_centroid_kmedoids as _sample_centroid_kmedoids_list,
    sample_centroid_voxel as _sample_centroid_voxel_list,
    sample_centroid_stratified as _sample_centroid_stratified_list,
    sample_uniform as _sample_uniform_list,
    # Sampling - NumPy API
    sample_centroid_uniform_numpy as _sample_centroid_uniform_numpy,
    sample_centroid_fps_numpy as _sample_centroid_fps_numpy,
    sample_centroid_kmedoids_numpy as _sample_centroid_kmedoids_numpy,
    sample_centroid_voxel_numpy as _sample_centroid_voxel_numpy,
    sample_centroid_stratified_numpy as _sample_centroid_stratified_numpy,
    sample_uniform_numpy as _sample_uniform_numpy,
)

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============ Size Measures ============


def bbox_volume(points):
    """Compute the n-dimensional bounding box hypervolume.

    Args:
        points: Either a list of lists or a numpy array (n_points × n_dims)

    Returns:
        The volume of the axis-aligned bounding box
    """
    if HAS_NUMPY and isinstance(points, np.ndarray):
        return _bbox_volume_numpy(points)
    return _bbox_volume_list(points)


def hull_volume(points):
    """Compute the convex hull hypervolume using QHull.

    Args:
        points: Either a list of lists or a numpy array (n_points × n_dims)

    Returns:
        The volume of the convex hull. Returns 0.0 for degenerate cases
        (insufficient points, coplanar points in higher dimensions, etc.)
    """
    if HAS_NUMPY and isinstance(points, np.ndarray):
        return _hull_volume_numpy(points)
    return _hull_volume_list(points)


def covariance_volume(points):
    """Compute covariance determinant as n-D ellipsoid volume proxy.

    Args:
        points: Either a list of lists or a numpy array (n_points × n_dims)

    Returns:
        The square root of the covariance determinant
    """
    if HAS_NUMPY and isinstance(points, np.ndarray):
        return _covariance_volume_numpy(points)
    return _covariance_volume_list(points)


def mean_dispersion(points):
    """Compute mean pairwise Euclidean distance between all points.

    Args:
        points: Either a list of lists or a numpy array (n_points × n_dims)

    Returns:
        The mean pairwise distance
    """
    if HAS_NUMPY and isinstance(points, np.ndarray):
        return _mean_dispersion_numpy(points)
    return _mean_dispersion_list(points)


# ============ Sampling Methods ============


def sample_centroid_uniform(points, count, seed):
    """Sample points using centroid + uniform random strategy.

    Args:
        points: Either a list of lists or a numpy array (n_points × n_dims)
        count: Number of points to sample
        seed: Random seed for reproducibility

    Returns:
        List of indices of sampled points
    """
    if HAS_NUMPY and isinstance(points, np.ndarray):
        return _sample_centroid_uniform_numpy(points, count, seed)
    return _sample_centroid_uniform_list(points, count, seed)


def sample_centroid_fps(points, count, seed):
    """Sample points using centroid + farthest point sampling strategy.

    Args:
        points: Either a list of lists or a numpy array (n_points × n_dims)
        count: Number of points to sample
        seed: Random seed for reproducibility

    Returns:
        List of indices of sampled points
    """
    if HAS_NUMPY and isinstance(points, np.ndarray):
        return _sample_centroid_fps_numpy(points, count, seed)
    return _sample_centroid_fps_list(points, count, seed)


def sample_centroid_kmedoids(points, count, seed, max_iterations=100):
    """Sample points using centroid + k-medoids strategy.

    Args:
        points: Either a list of lists or a numpy array (n_points × n_dims)
        count: Number of points to sample
        seed: Random seed for reproducibility
        max_iterations: Maximum iterations for k-medoids optimization (default: 100)

    Returns:
        List of indices of sampled points
    """
    if HAS_NUMPY and isinstance(points, np.ndarray):
        return _sample_centroid_kmedoids_numpy(points, count, seed, max_iterations)
    return _sample_centroid_kmedoids_list(points, count, seed, max_iterations)


def sample_centroid_voxel(points, count, seed, voxel_size):
    """Sample points using centroid + voxel grid strategy.

    Args:
        points: Either a list of lists or a numpy array (n_points × n_dims)
        count: Number of points to sample
        seed: Random seed for reproducibility
        voxel_size: Size of voxel grid cells

    Returns:
        List of indices of sampled points
    """
    if HAS_NUMPY and isinstance(points, np.ndarray):
        return _sample_centroid_voxel_numpy(points, count, seed, voxel_size)
    return _sample_centroid_voxel_list(points, count, seed, voxel_size)


def sample_centroid_stratified(points, count, seed, num_strata, max_iterations=100):
    """Sample points using centroid + stratified sampling strategy.

    Args:
        points: Either a list of lists or a numpy array (n_points × n_dims)
        count: Number of points to sample
        seed: Random seed for reproducibility
        num_strata: Number of strata (clusters) for stratification
        max_iterations: Maximum iterations for k-means clustering (default: 100)

    Returns:
        List of indices of sampled points
    """
    if HAS_NUMPY and isinstance(points, np.ndarray):
        return _sample_centroid_stratified_numpy(
            points, count, seed, num_strata, max_iterations
        )
    return _sample_centroid_stratified_list(
        points, count, seed, num_strata, max_iterations
    )


def sample_uniform(points, count, seed):
    """Sample points using pure uniform random strategy (no centroid guarantee).

    Args:
        points: Either a list of lists or a numpy array (n_points × n_dims)
        count: Number of points to sample
        seed: Random seed for reproducibility

    Returns:
        List of indices of sampled points
    """
    if HAS_NUMPY and isinstance(points, np.ndarray):
        return _sample_uniform_numpy(points, count, seed)
    return _sample_uniform_list(points, count, seed)


# ============ Cluster sampling module ============ 

from gasp.clustersampler import (
    sample_from_clusters,
    grid_search_sample_from_clusters,
    SIZE_MEASURES as CLUSTER_SIZE_MEASURES,
    SAMPLING_METHODS as CLUSTER_SAMPLING_METHODS,
)

__all__ = [
    # Size measures
    "bbox_volume",
    "hull_volume",
    "covariance_volume",
    "mean_dispersion",
    # Sampling methods
    "sample_centroid_uniform",
    "sample_centroid_fps",
    "sample_centroid_kmedoids",
    "sample_centroid_voxel",
    "sample_centroid_stratified",
    "sample_uniform",
    # Cluster sampling
    "sample_from_clusters",
    "grid_search_sample_from_clusters",
    "CLUSTER_SIZE_MEASURES",
    "CLUSTER_SAMPLING_METHODS",
]


def main() -> None:
    """Demo function showing all available size measures and sampling methods."""
    print("=== GASP Demo ===\n")

    # Simple 2D example
    print("2D Triangle (lists):")
    points_2d = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.866],
    ]
    print(f"  Points: {points_2d}")
    print(f"  Bounding box volume: {bbox_volume(points_2d):.4f}")
    print(f"  Convex hull volume: {hull_volume(points_2d):.4f}")
    print(f"  Covariance volume: {covariance_volume(points_2d):.4f}")
    print(f"  Mean dispersion: {mean_dispersion(points_2d):.4f}")

    if HAS_NUMPY:
        print("\n2D Triangle (numpy):")
        points_2d_np = np.array(points_2d)
        print(f"  Array shape: {points_2d_np.shape}")
        print(f"  Bounding box volume: {bbox_volume(points_2d_np):.4f}")
        print(f"  Convex hull volume: {hull_volume(points_2d_np):.4f}")
        print(f"  Covariance volume: {covariance_volume(points_2d_np):.4f}")
        print(f"  Mean dispersion: {mean_dispersion(points_2d_np):.4f}")

    # 3D example
    print("\n3D Tetrahedron:")
    points_3d = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    print("  Points: 4 vertices of unit tetrahedron")
    print(f"  Bounding box volume: {bbox_volume(points_3d):.4f}")
    print(f"  Convex hull volume: {hull_volume(points_3d):.6f}")  # Should be 1/6
    print(f"  Covariance volume: {covariance_volume(points_3d):.4f}")
    print(f"  Mean dispersion: {mean_dispersion(points_3d):.4f}")

    if HAS_NUMPY:
        # Sampling demo
        print("\n\nSampling Demo (5D Random Point Cloud):")
        np.random.seed(42)
        points_5d = np.random.randn(100, 5)
        print(f"  Original points: {points_5d.shape}")

        # Try different sampling methods
        print("\n  Uniform sampling (10 points):")
        indices = sample_uniform(points_5d, 10, seed=42)
        print(f"    Sampled indices: {indices[:5]}... ({len(indices)} total)")

        print("\n  Centroid + FPS (10 points):")
        indices = sample_centroid_fps(points_5d, 10, seed=42)
        print(f"    Sampled indices: {indices[:5]}... ({len(indices)} total)")

        print("\n  Centroid + Stratified (10 points, 3 strata):")
        indices = sample_centroid_stratified(points_5d, 10, seed=42, num_strata=3)
        print(f"    Sampled indices: {indices[:5]}... ({len(indices)} total)")

        # Higher dimensional example
        print("\n5D Random Point Cloud - Size Measures:")
        print(f"  Array shape: {points_5d.shape}")
        print(f"  Bounding box volume: {bbox_volume(points_5d):.4f}")
        print(f"  Convex hull volume: {hull_volume(points_5d):.4f}")
        print(f"  Covariance volume: {covariance_volume(points_5d):.4f}")
        print(f"  Mean dispersion: {mean_dispersion(points_5d):.4f}")


if __name__ == "__main__":
    main()
