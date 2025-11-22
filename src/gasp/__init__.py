from gasp._core import (
    bbox_volume as _bbox_volume_list,
    hull_volume as _hull_volume_list,
    covariance_volume as _covariance_volume_list,
    mean_dispersion as _mean_dispersion_list,
    bbox_volume_numpy as _bbox_volume_numpy,
    hull_volume_numpy as _hull_volume_numpy,
    covariance_volume_numpy as _covariance_volume_numpy,
    mean_dispersion_numpy as _mean_dispersion_numpy,
)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


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


__all__ = [
    "max_value",
    "bbox_volume",
    "hull_volume",
    "covariance_volume",
    "mean_dispersion",
]


def main() -> None:
    """Demo function showing all available size measures."""
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
    print(f"  Points: 4 vertices of unit tetrahedron")
    print(f"  Bounding box volume: {bbox_volume(points_3d):.4f}")
    print(f"  Convex hull volume: {hull_volume(points_3d):.6f}")  # Should be 1/6
    print(f"  Covariance volume: {covariance_volume(points_3d):.4f}")
    print(f"  Mean dispersion: {mean_dispersion(points_3d):.4f}")
    
    if HAS_NUMPY:
        # Higher dimensional example with random data
        print("\n5D Random Point Cloud (numpy):")
        np.random.seed(42)
        points_5d = np.random.randn(100, 5)
        print(f"  Array shape: {points_5d.shape}")
        print(f"  Bounding box volume: {bbox_volume(points_5d):.4f}")
        print(f"  Convex hull volume: {hull_volume(points_5d):.4f}")
        print(f"  Covariance volume: {covariance_volume(points_5d):.4f}")
        print(f"  Mean dispersion: {mean_dispersion(points_5d):.4f}")
    

if __name__ == "__main__":
    main()
