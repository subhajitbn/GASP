import numpy as np
from scipy.spatial import ConvexHull

# ----------------------------
# GASP: Geometric Allocation of Sample Points
# ----------------------------

def size_bbox(points):
    """Bounding box volume"""
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return np.prod(maxs - mins)

def size_hull(points):
    """Convex hull volume"""
    if len(points) < 4:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def size_covdet(points):
    """Covariance determinant (ellipsoid volume proxy)"""
    cov = np.cov(points.T)
    return np.sqrt(np.linalg.det(cov + 1e-8 * np.eye(3)))  # stability

def size_dispersion(points):
    """Mean pairwise distance"""
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return np.mean(dists)

SIZE_MEASURES = {
    "bbox": size_bbox,
    "hull": size_hull,
    "covdet": size_covdet,
    "dispersion": size_dispersion
}

def gasp_allocate(point_clouds, total_points=100, measure="hull"):
    """Allocate sampling quota to each point cloud based on geometric measure"""
    f = SIZE_MEASURES.get(measure)
    if f is None:
        raise ValueError(f"Unknown measure '{measure}'. Available: {list(SIZE_MEASURES.keys())}")

    sizes = np.array([f(pc) for pc in point_clouds])
    sizes = sizes / np.sum(sizes)
    allocations = np.maximum(1, np.round(total_points * sizes)).astype(int)

    # Adjust to ensure sum == total_points
    diff = total_points - np.sum(allocations)
    if diff != 0:
        # distribute excess/deficit greedily
        order = np.argsort(sizes)[::-1 if diff > 0 else 1]
        for i in order:
            if diff == 0:
                break
            allocations[i] += np.sign(diff)
            diff -= np.sign(diff)

    return allocations, sizes

# ----------------------------
# Dummy Dataset (10 point clouds)
# ----------------------------
def generate_dummy_pointclouds():
    np.random.seed(42)
    clouds = []
    for i in range(10):
        n_points = np.random.randint(50, 300)
        scale = np.random.uniform(0.1, 2.0)  # controls spread
        center = np.random.uniform(-5, 5, size=(3,))
        points = np.random.randn(n_points, 3) * scale + center
        clouds.append(points)
    return clouds

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    point_clouds = generate_dummy_pointclouds()
    total_quota = 100

    for measure in SIZE_MEASURES.keys():
        alloc, sizes = gasp_allocate(point_clouds, total_quota, measure)
        print(f"\nMeasure: {measure}")
        for i, (k, s) in enumerate(zip(alloc, sizes)):
            print(f"Cloud {i+1:2d}: size={s:.4f}, quota={k}")
        print(f"Total allocated: {np.sum(alloc)}\n")

