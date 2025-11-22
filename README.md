# GASP: Geometric Analysis and Sampling Package

High-performance proportional sampling from point cloud clusters with geometric size measures and sampling measures.
## Installation

```bash
pip install gasp-python
```

## Features

### Proportional Cluster Sampling
Sample from multiple point cloud clusters with automatic proportional allocation based on geometric size measures:

```python
from gasp import sample_from_clusters

# Sample 1000 points proportionally from clusters
# Each cluster gets samples proportional to its convex hull volume
sampled = sample_from_clusters(
    directory="./my_clusters",  # Directory with .npy files
    total_samples=1000,
    size_measure="hull_volume",
    sampling_method="centroid_fps",
    seed=42
)

# Returns: {cluster_name: sampled_points_array, ...}
for name, points in sampled.items():
    print(f"{name}: sampled {len(points)} points")
```

### Geometric Size Measures
- `bbox_volume` - Axis-aligned bounding box hypervolume
- `hull_volume` - Convex hull hypervolume  
- `covariance_volume` - Covariance determinant (ellipsoid proxy)
- `mean_dispersion` - Mean pairwise Euclidean distance

### Sampling Methods
Multiple strategies for representative point selection within each cluster:
- `uniform` - Pure random sampling
- `centroid_uniform` - Guarantees centroid + random
- `centroid_fps` - Farthest point sampling
- `centroid_kmedoids` - K-medoids clustering
- `centroid_voxel` - Voxel grid sampling
- `centroid_stratified` - Stratified sampling

### Grid Search
Test multiple configurations:

```python
from gasp import grid_search_sample_from_clusters

results = grid_search_sample_from_clusters(
    directory="./my_clusters",
    total_samples=1000,
    size_measures=["bbox_volume", "hull_volume"],
    sampling_methods=["uniform", "centroid_fps"],
    seed=42
)

# Access specific configuration
best_result = results[("hull_volume", "centroid_fps")]
```

## Direct Point Cloud Operations

You can also use the low-level functions directly:

```python
import numpy as np
from gasp import bbox_volume, hull_volume, sample_centroid_fps

# Single point cloud
points = np.random.randn(1000, 3)

# Compute geometric measures
volume = bbox_volume(points)
hull_vol = hull_volume(points)

# Sample representative points
indices = sample_centroid_fps(points, count=100, seed=42)
sampled = points[indices]
```

## How It Works

1. **Load clusters**: Reads all `.npy` files from a directory
2. **Compute sizes**: Measures each cluster using the specified size measure
3. **Proportional allocation**: Distributes N samples across clusters proportionally (minimum 1 per cluster)
4. **Sample points**: Uses the specified sampling method within each cluster

## Performance

- **Fast**: Rust core for high-performance computation
- **Flexible**: Works with Python lists or NumPy arrays
- **N-Dimensional**: Supports 2D and 3D. Work ongoing for N-dim support.
- **Robust**: Handles edge cases (empty clusters, degenerate geometries)