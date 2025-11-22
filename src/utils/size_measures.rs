use nalgebra::{DMatrix, DVector, Point2, Point3};
use parry2d::shape::{ConvexPolygon, Shape as Shape2d};
use parry3d::shape::{ConvexPolyhedron, Shape as Shape3d};

/// Compute n-dimensional bounding box hypervolume.
pub fn size_bbox(points: &[DVector<f64>]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let dim = points[0].len();
    let mut min_v = points[0].clone();
    let mut max_v = points[0].clone();
    for p in points.iter() {
        for d in 0..dim {
            if p[d] < min_v[d] {
                min_v[d] = p[d];
            }
            if p[d] > max_v[d] {
                max_v[d] = p[d];
            }
        }
    }
    (0..dim)
        .map(|d| max_v[d] - min_v[d])
        .fold(1.0, |acc, x| acc * x.abs())
}

/// Compute convex hull hypervolume using parry2d/parry3d.
///
/// Returns 0.0 for edge cases:
/// - Empty point set
/// - Single point (0D hull)
/// - Degenerate configurations (coplanar points in higher dimensions)
/// - Insufficient points for dimension (need at least dim+1 points for a simplex)
/// - Dimensions other than 2D or 3D (not supported)
pub fn size_hull(points: &[DVector<f64>]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    let dim = points[0].len();

    // Need at least dim+1 points for a simplex
    if points.len() <= dim {
        return 0.0;
    }

    match dim {
        2 => size_hull_2d(points),
        3 => size_hull_3d(points),
        _ => 0.0, // Unsupported dimensions
    }
}

fn size_hull_2d(points: &[DVector<f64>]) -> f64 {
    let pts: Vec<Point2<f32>> = points
        .iter()
        .map(|p| Point2::new(p[0] as f32, p[1] as f32))
        .collect();

    ConvexPolygon::from_convex_hull(&pts)
        .and_then(|poly| {
            // Compute mass properties with density 1.0
            let mass_props = poly.mass_properties(1.0);
            Some(mass_props.mass() as f64)
        })
        .unwrap_or(0.0)
}

fn size_hull_3d(points: &[DVector<f64>]) -> f64 {
    let pts: Vec<Point3<f32>> = points
        .iter()
        .map(|p| Point3::new(p[0] as f32, p[1] as f32, p[2] as f32))
        .collect();

    ConvexPolyhedron::from_convex_hull(&pts)
        .and_then(|hull| {
            // Compute mass properties with density 1.0
            let mass_props = hull.mass_properties(1.0);
            Some(mass_props.mass() as f64)
        })
        .unwrap_or(0.0)
}

/// Covariance determinant as nD ellipsoid volume proxy.
pub fn size_covdet(points: &[DVector<f64>]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let n = points.len() as f64;
    let dim = points[0].len();

    // Compute mean
    let mut mean = DVector::zeros(dim);
    for p in points {
        mean += p;
    }
    mean /= n;

    // Build data matrix (dim × n)
    let mut data = DMatrix::zeros(dim, points.len());
    for (i, p) in points.iter().enumerate() {
        data.set_column(i, &(p - &mean));
    }

    // Covariance = (1/(n−1)) * data * dataᵀ
    let cov = (&data * data.transpose()) / (n - 1.0);

    // Add small regularization for numerical stability
    let cov_reg = &cov + DMatrix::identity(dim, dim) * 1e-8;

    // Return square root of determinant
    cov_reg.determinant().abs().sqrt()
}

/// Mean pairwise distance (O(n²)), valid in any dimension.
pub fn size_dispersion(points: &[DVector<f64>]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let mut total_dist = 0.0;
    let mut count = 0;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            total_dist += (&points[i] - &points[j]).norm();
            count += 1;
        }
    }
    total_dist / (count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bbox_2d_unit_square() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![1.0, 1.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        assert_relative_eq!(size_bbox(&points), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hull_2d_triangle() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        // Area of right triangle = 0.5 * base * height = 0.5 * 1 * 1
        let volume = size_hull(&points);
        assert!(volume > 0.0, "Hull volume should be positive");
        assert_relative_eq!(volume, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_hull_3d_tetrahedron() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0, 0.0]),
            DVector::from_vec(vec![0.0, 0.0, 1.0]),
        ];
        // Volume of tetrahedron = 1/6
        let volume = size_hull(&points);
        assert!(volume > 0.0, "Hull volume should be positive");
        assert_relative_eq!(volume, 1.0 / 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hull_insufficient_points() {
        // 2 points in 3D - can't form a 3D hull
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0, 0.0]),
            DVector::from_vec(vec![1.0, 1.0, 1.0]),
        ];
        assert_eq!(size_hull(&points), 0.0);
    }

    #[test]
    fn test_hull_degenerate_coplanar() {
        // 4 points in 3D that are coplanar (all z=0)
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0, 0.0]),
            DVector::from_vec(vec![1.0, 1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0, 0.0]),
        ];
        // Should return 0 since they don't span 3D space
        assert_eq!(size_hull(&points), 0.0);
    }

    #[test]
    fn test_empty_points() {
        let points: Vec<DVector<f64>> = vec![];
        assert_eq!(size_bbox(&points), 0.0);
        assert_eq!(size_hull(&points), 0.0);
        assert_eq!(size_covdet(&points), 0.0);
        assert_eq!(size_dispersion(&points), 0.0);
    }

    #[test]
    fn test_covdet_basic() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        // Should be non-zero for non-degenerate points
        assert!(size_covdet(&points) > 0.0);
    }

    #[test]
    fn test_dispersion_basic() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
        ];
        // Distance between two points is 1.0
        assert_relative_eq!(size_dispersion(&points), 1.0, epsilon = 1e-10);
    }
}
