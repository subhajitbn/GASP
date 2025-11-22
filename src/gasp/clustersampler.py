"""
Cluster Sampling Module
Samples points from clustered point clouds with proportional allocation.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Callable, Optional
import numpy as np

# Import directly from _core to avoid circular imports
from gasp._core import (
    bbox_volume_numpy,
    hull_volume_numpy,
    covariance_volume_numpy,
    mean_dispersion_numpy,
    sample_uniform_numpy,
    sample_centroid_uniform_numpy,
    sample_centroid_fps_numpy,
    sample_centroid_kmedoids_numpy,
    sample_centroid_voxel_numpy,
    sample_centroid_stratified_numpy,
)


# Size measure functions (using numpy variants directly since we convert to float64)
SIZE_MEASURES = {
    "bbox_volume": bbox_volume_numpy,
    "hull_volume": hull_volume_numpy,
    "covariance_volume": covariance_volume_numpy,
    "mean_dispersion": mean_dispersion_numpy,
}

# Sampling functions (with their additional parameters)
SAMPLING_METHODS = {
    "uniform": lambda points, count, seed: sample_uniform_numpy(points, count, seed),
    "centroid_uniform": lambda points, count, seed: sample_centroid_uniform_numpy(
        points, count, seed
    ),
    "centroid_fps": lambda points, count, seed: sample_centroid_fps_numpy(
        points, count, seed
    ),
    "centroid_kmedoids": lambda points, count, seed: sample_centroid_kmedoids_numpy(
        points, count, seed, max_iterations=100
    ),
    "centroid_voxel": lambda points, count, seed: sample_centroid_voxel_numpy(
        points, count, seed, voxel_size=0.1
    ),
    "centroid_stratified": lambda points, count, seed: sample_centroid_stratified_numpy(
        points, count, seed, num_strata=max(2, count // 3), max_iterations=100
    ),
}


def load_clusters(directory: Path) -> List[Tuple[str, np.ndarray]]:
    """Load all .npy files from directory as clusters.
    
    Args:
        directory: Path to directory containing .npy files
        
    Returns:
        List of (filename, points_array) tuples
    """
    clusters = []
    npy_files = sorted(Path(directory).glob("*.npy"))
    
    if not npy_files:
        raise ValueError(f"No .npy files found in {directory}")
    
    for npy_file in npy_files:
        points = np.load(npy_file)
        if points.ndim != 2:
            print(f"Warning: Skipping {npy_file.name} - expected 2D array, got shape {points.shape}")
            continue
        # Convert to float64 to ensure compatibility with GASP
        points = np.asarray(points, dtype=np.float64)
        clusters.append((npy_file.name, points))
    
    if not clusters:
        raise ValueError("No valid cluster files loaded")
    
    return clusters


def compute_cluster_sizes(
    clusters: List[Tuple[str, np.ndarray]], size_measure: Callable
) -> List[float]:
    """Compute size of each cluster using the specified measure.
    
    Args:
        clusters: List of (filename, points) tuples
        size_measure: Size measurement function
        
    Returns:
        List of size values for each cluster
    """
    sizes = []
    for name, points in clusters:
        try:
            size = size_measure(points)
            # Handle edge cases (zero or negative sizes)
            if size <= 0:
                size = 1e-10  # Small positive value
            sizes.append(size)
        except Exception as e:
            print(f"Warning: Failed to compute size for {name}: {e}")
            sizes.append(1e-10)
    
    return sizes


def allocate_samples(
    sizes: List[float], 
    total_samples: int, 
    num_clusters: int,
    cluster_point_counts: Optional[List[int]] = None
) -> List[int]:
    """Allocate samples proportionally with minimum 1 per cluster.
    
    Args:
        sizes: Size of each cluster
        total_samples: Total number N of samples to allocate
        num_clusters: Number of clusters
        cluster_point_counts: Optional list of actual point counts per cluster
                             for handling capacity constraints
        
    Returns:
        List of sample counts for each cluster
    """
    if total_samples < num_clusters:
        raise ValueError(
            f"N={total_samples} is less than number of clusters ({num_clusters}). "
            "Each cluster needs at least 1 sample."
        )
    
    # Start with 1 sample per cluster
    allocation = [1] * num_clusters
    remaining = total_samples - num_clusters
    
    if remaining == 0:
        return allocation
    
    # Distribute remaining samples proportionally
    total_size = sum(sizes)
    if total_size == 0:
        # Fallback: distribute evenly
        for i in range(remaining):
            allocation[i % num_clusters] += 1
        return allocation
    
    # Calculate proportional allocation
    proportions = [size / total_size for size in sizes]
    fractional_allocation = [p * remaining for p in proportions]
    
    # Allocate integer parts
    for i, frac in enumerate(fractional_allocation):
        allocation[i] += int(frac)
    
    # Distribute remainder using largest fractional parts
    allocated = sum(allocation)
    remainders = [(frac - int(frac), i) for i, frac in enumerate(fractional_allocation)]
    remainders.sort(reverse=True)
    
    for i in range(total_samples - allocated):
        _, idx = remainders[i]
        allocation[idx] += 1
    
    # Handle capacity constraints if cluster sizes are provided
    if cluster_point_counts is not None:
        # Cap allocations and collect overflow
        overflow = 0
        for i in range(num_clusters):
            if allocation[i] > cluster_point_counts[i]:
                overflow += allocation[i] - cluster_point_counts[i]
                allocation[i] = cluster_point_counts[i]
        
        # Redistribute overflow to clusters that can take more
        while overflow > 0:
            # Find clusters that can accept more points, sorted by size
            available = [
                (sizes[i], i) for i in range(num_clusters)
                if allocation[i] < cluster_point_counts[i]
            ]
            
            if not available:
                # No more capacity, we'll return fewer samples than requested
                print(f"Warning: Could only allocate {sum(allocation)} samples out of {total_samples} requested")
                break
            
            available.sort(reverse=True)  # Prioritize larger clusters
            
            # Give overflow to the largest cluster with capacity
            _, idx = available[0]
            give = min(overflow, cluster_point_counts[idx] - allocation[idx])
            allocation[idx] += give
            overflow -= give
    
    return allocation


def sample_clusters(
    clusters: List[Tuple[str, np.ndarray]],
    allocation: List[int],
    sampling_method: Callable,
    seed: int,
) -> Dict[str, List[int]]:
    """Sample points from each cluster according to allocation.
    
    Args:
        clusters: List of (filename, points) tuples
        allocation: Number of samples for each cluster
        sampling_method: Sampling function
        seed: Random seed
        
    Returns:
        Dictionary mapping cluster names to sampled indices
    """
    results = {}
    
    for i, ((name, points), count) in enumerate(zip(clusters, allocation)):
        try:
            # Ensure we don't request more samples than points available
            count = min(count, len(points))
            
            if count == 0:
                results[name] = []
            elif count >= len(points):
                # Take all points
                results[name] = list(range(len(points)))
            else:
                # Use sampling method with unique seed per cluster
                indices = sampling_method(points, count, seed + i)
                results[name] = indices
        except Exception as e:
            print(f"Warning: Failed to sample from {name}: {e}")
            # Fallback: random sampling
            indices = np.random.RandomState(seed + i).choice(
                len(points), size=min(count, len(points)), replace=False
            ).tolist()
            results[name] = indices
    
    return results


def sample_from_clusters(
    directory: str,
    total_samples: int,
    size_measure: str = "bbox_volume",
    sampling_method: str = "centroid_fps",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Sample points from clustered point clouds with proportional allocation.
    
    This function loads all .npy files from a directory (each representing a cluster),
    computes their sizes using the specified measure, allocates samples proportionally
    (with at least 1 point per cluster), and returns the sampled points.
    
    Args:
        directory: Path to directory containing .npy cluster files
        total_samples: Total number N of samples to draw across all clusters
        size_measure: Size measure to use for proportional allocation.
                     Options: "bbox_volume", "hull_volume", "covariance_volume", "mean_dispersion"
        sampling_method: Sampling strategy to use within each cluster.
                        Options: "uniform", "centroid_uniform", "centroid_fps", 
                                "centroid_kmedoids", "centroid_voxel", "centroid_stratified"
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping cluster names (filenames) to sampled point arrays (n_sampled × n_dims)
        
    Raises:
        ValueError: If directory is invalid, no clusters found, or total_samples < num_clusters
        KeyError: If size_measure or sampling_method is not recognized
        
    Example:
        >>> import gasp
        >>> sampled = gasp.sample_from_clusters(
        ...     directory="./my_clusters",
        ...     total_samples=1000,
        ...     size_measure="hull_volume",
        ...     sampling_method="centroid_fps",
        ...     seed=42
        ... )
        >>> for cluster_name, points in sampled.items():
        ...     print(f"{cluster_name}: {points.shape}")
    """
    # Validate inputs
    if size_measure not in SIZE_MEASURES:
        raise KeyError(
            f"Unknown size measure: {size_measure}. "
            f"Available options: {list(SIZE_MEASURES.keys())}"
        )
    
    if sampling_method not in SAMPLING_METHODS:
        raise KeyError(
            f"Unknown sampling method: {sampling_method}. "
            f"Available options: {list(SAMPLING_METHODS.keys())}"
        )
    
    # Load clusters
    clusters = load_clusters(Path(directory))
    
    # Get point counts for capacity constraint handling
    cluster_point_counts = [len(points) for _, points in clusters]
    
    # Compute sizes
    size_fn = SIZE_MEASURES[size_measure]
    sizes = compute_cluster_sizes(clusters, size_fn)
    
    # Allocate samples with capacity constraints
    allocation = allocate_samples(sizes, total_samples, len(clusters), cluster_point_counts)
    
    # Sample from clusters (get indices)
    sampling_fn = SAMPLING_METHODS[sampling_method]
    sampled_indices = sample_clusters(clusters, allocation, sampling_fn, seed)
    
    # Convert indices to actual point arrays
    result = {}
    for (name, points), indices in zip(clusters, sampled_indices.values()):
        if len(indices) > 0:
            result[name] = points[indices]
        else:
            # Empty array with correct shape
            result[name] = np.empty((0, points.shape[1]), dtype=points.dtype)
    
    return result


def grid_search_sample_from_clusters(
    directory: str,
    total_samples: int,
    size_measures: Optional[List[str]] = None,
    sampling_methods: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    """
    Run grid search over multiple size measures and sampling methods.
    
    Args:
        directory: Path to directory containing .npy cluster files
        total_samples: Total number N of samples to draw across all clusters
        size_measures: List of size measures to try (None = all available)
        sampling_methods: List of sampling methods to try (None = all available)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping (size_measure, sampling_method) tuples to sampled results.
        Each result is a dict mapping cluster names to sampled point arrays.
        
    Example:
        >>> import gasp
        >>> results = gasp.grid_search_sample_from_clusters(
        ...     directory="./my_clusters",
        ...     total_samples=1000,
        ...     size_measures=["bbox_volume", "hull_volume"],
        ...     sampling_methods=["uniform", "centroid_fps"],
        ...     seed=42
        ... )
        >>> # Access specific configuration
        >>> bbox_fps_result = results[("bbox_volume", "centroid_fps")]
    """
    if size_measures is None:
        size_measures = list(SIZE_MEASURES.keys())
    if sampling_methods is None:
        sampling_methods = list(SAMPLING_METHODS.keys())
    
    results = {}
    
    for size_measure in size_measures:
        for sampling_method in sampling_methods:
            result = sample_from_clusters(
                directory=directory,
                total_samples=total_samples,
                size_measure=size_measure,
                sampling_method=sampling_method,
                seed=seed,
            )
            results[(size_measure, sampling_method)] = result
    
    return results


__all__ = [
    "sample_from_clusters",
    "grid_search_sample_from_clusters",
    "SIZE_MEASURES",
    "SAMPLING_METHODS",
]
