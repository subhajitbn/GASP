use nalgebra::DVector;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{HashMap, HashSet};

/// Sampling strategy trait - each method takes points and returns indices to sample
pub trait SamplingStrategy {
    fn sample(&self, points: &[DVector<f64>], count: usize, rng: &mut impl Rng) -> Vec<usize>;
    fn name(&self) -> &str;
}

/// Public API functions for Python bindings

/// Centroid + Uniform Random sampling
pub fn sample_centroid_uniform(points: &[DVector<f64>], count: usize, seed: u64) -> Vec<usize> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let sampler = CentroidPlusUniformRandom;
    sampler.sample(points, count, &mut rng)
}

/// Centroid + Farthest Point Sampling
pub fn sample_centroid_fps(points: &[DVector<f64>], count: usize, seed: u64) -> Vec<usize> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let sampler = CentroidPlusFPS;
    sampler.sample(points, count, &mut rng)
}

/// Centroid + k-Medoids sampling
pub fn sample_centroid_kmedoids(
    points: &[DVector<f64>],
    count: usize,
    seed: u64,
    max_iterations: usize,
) -> Vec<usize> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let sampler = CentroidPlusKMedoids::new(max_iterations);
    sampler.sample(points, count, &mut rng)
}

/// Centroid + Voxel Grid sampling
pub fn sample_centroid_voxel(
    points: &[DVector<f64>],
    count: usize,
    seed: u64,
    voxel_size: f64,
) -> Vec<usize> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let sampler = CentroidPlusVoxelGrid::new(voxel_size);
    sampler.sample(points, count, &mut rng)
}

/// Centroid + Stratified sampling
pub fn sample_centroid_stratified(
    points: &[DVector<f64>],
    count: usize,
    seed: u64,
    num_strata: usize,
    max_iterations: usize,
) -> Vec<usize> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let sampler = CentroidPlusStratified::new(num_strata, max_iterations);
    sampler.sample(points, count, &mut rng)
}

/// Pure Uniform Random sampling (no centroid guarantee)
pub fn sample_uniform(points: &[DVector<f64>], count: usize, seed: u64) -> Vec<usize> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let sampler = UniformRandom;
    sampler.sample(points, count, &mut rng)
}

/// Helper: compute centroid of points
fn compute_centroid(points: &[DVector<f64>]) -> DVector<f64> {
    if points.is_empty() {
        return DVector::zeros(0);
    }
    let mut sum = DVector::zeros(points[0].len());
    for p in points {
        sum += p;
    }
    sum / (points.len() as f64)
}

/// Find index of point closest to target
fn find_closest_point(points: &[DVector<f64>], target: &DVector<f64>) -> usize {
    points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let dist_a = (*a - target).norm();
            let dist_b = (*b - target).norm();
            dist_a.partial_cmp(&dist_b).unwrap()
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// 1. Centroid + Uniform Random
pub struct CentroidPlusUniformRandom;

impl SamplingStrategy for CentroidPlusUniformRandom {
    fn sample(&self, points: &[DVector<f64>], count: usize, rng: &mut impl Rng) -> Vec<usize> {
        if points.is_empty() || count == 0 {
            return vec![];
        }

        let count = count.min(points.len());
        let mut selected = HashSet::new();

        // Always include centroid (closest point to it)
        let centroid = compute_centroid(points);
        let centroid_idx = find_closest_point(points, &centroid);
        selected.insert(centroid_idx);

        // Fill rest with uniform random
        let mut indices: Vec<usize> = (0..points.len()).collect();
        indices.shuffle(rng);

        for idx in indices {
            if selected.len() >= count {
                break;
            }
            selected.insert(idx);
        }

        selected.into_iter().collect()
    }

    fn name(&self) -> &str {
        "CentroidPlusUniformRandom"
    }
}

/// 2. Centroid + Farthest Point Sampling (FPS)
pub struct CentroidPlusFPS;

impl SamplingStrategy for CentroidPlusFPS {
    fn sample(&self, points: &[DVector<f64>], count: usize, _rng: &mut impl Rng) -> Vec<usize> {
        if points.is_empty() || count == 0 {
            return vec![];
        }

        let count = count.min(points.len());
        let mut selected = Vec::new();
        let mut min_distances = vec![f64::INFINITY; points.len()];

        // Start with centroid
        let centroid = compute_centroid(points);
        let centroid_idx = find_closest_point(points, &centroid);
        selected.push(centroid_idx);

        // Update distances from centroid
        for i in 0..points.len() {
            let dist = (&points[i] - &points[centroid_idx]).norm();
            min_distances[i] = dist;
        }

        // Greedily select farthest points
        for _ in 1..count {
            let farthest_idx = min_distances
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            selected.push(farthest_idx);

            // Update distances
            for i in 0..points.len() {
                let dist = (&points[i] - &points[farthest_idx]).norm();
                min_distances[i] = min_distances[i].min(dist);
            }
        }

        selected
    }

    fn name(&self) -> &str {
        "CentroidPlusFPS"
    }
}

/// 3. Centroid + k-medoids (simplified PAM)
pub struct CentroidPlusKMedoids {
    max_iterations: usize,
}

impl CentroidPlusKMedoids {
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }
}

impl SamplingStrategy for CentroidPlusKMedoids {
    fn sample(&self, points: &[DVector<f64>], count: usize, rng: &mut impl Rng) -> Vec<usize> {
        if points.is_empty() || count == 0 {
            return vec![];
        }

        let count = count.min(points.len());
        let mut medoids = Vec::new();

        // Start with centroid
        let centroid = compute_centroid(points);
        let centroid_idx = find_closest_point(points, &centroid);
        medoids.push(centroid_idx);

        // Initialize remaining medoids randomly
        let mut indices: Vec<usize> = (0..points.len()).filter(|&i| i != centroid_idx).collect();
        indices.shuffle(rng);
        for i in 0..(count - 1).min(indices.len()) {
            medoids.push(indices[i]);
        }

        // PAM iterations (simplified)
        for _ in 0..self.max_iterations {
            let mut improved = false;

            for med_idx in 1..medoids.len() {
                let current_medoid = medoids[med_idx];
                let current_cost = compute_total_distance(points, &medoids);

                // Try swapping with non-medoid points
                for candidate in 0..points.len() {
                    if medoids.contains(&candidate) {
                        continue;
                    }

                    medoids[med_idx] = candidate;
                    let new_cost = compute_total_distance(points, &medoids);

                    if new_cost < current_cost {
                        improved = true;
                        break;
                    } else {
                        medoids[med_idx] = current_medoid;
                    }
                }

                if improved {
                    break;
                }
            }

            if !improved {
                break;
            }
        }

        medoids
    }

    fn name(&self) -> &str {
        "CentroidPlusKMedoids"
    }
}

fn compute_total_distance(points: &[DVector<f64>], medoids: &[usize]) -> f64 {
    let mut total = 0.0;
    for i in 0..points.len() {
        let min_dist = medoids
            .iter()
            .map(|&m| (&points[i] - &points[m]).norm())
            .fold(f64::INFINITY, f64::min);
        total += min_dist;
    }
    total
}

/// 4. Centroid + Voxel Grid Sampling
pub struct CentroidPlusVoxelGrid {
    voxel_size: f64,
}

impl CentroidPlusVoxelGrid {
    pub fn new(voxel_size: f64) -> Self {
        Self { voxel_size }
    }
}

impl SamplingStrategy for CentroidPlusVoxelGrid {
    fn sample(&self, points: &[DVector<f64>], count: usize, rng: &mut impl Rng) -> Vec<usize> {
        if points.is_empty() || count == 0 {
            return vec![];
        }

        let count = count.min(points.len());
        let mut selected = HashSet::new();

        // Always include centroid
        let centroid = compute_centroid(points);
        let centroid_idx = find_closest_point(points, &centroid);
        selected.insert(centroid_idx);

        // Voxelize remaining points
        let dim = points[0].len();
        let mut voxel_map: HashMap<Vec<i32>, Vec<usize>> = HashMap::new();

        for (idx, point) in points.iter().enumerate() {
            let voxel_key: Vec<i32> = (0..dim)
                .map(|d| (point[d] / self.voxel_size).floor() as i32)
                .collect();
            voxel_map
                .entry(voxel_key)
                .or_insert_with(Vec::new)
                .push(idx);
        }

        // Sample from voxels
        let mut voxel_list: Vec<_> = voxel_map.into_iter().collect();
        voxel_list.shuffle(rng);

        for (_, mut indices) in voxel_list {
            if selected.len() >= count {
                break;
            }
            // Pick random point from this voxel
            if !indices.is_empty() {
                indices.shuffle(rng);
                selected.insert(indices[0]);
            }
        }

        // If still need more, add random points
        let mut remaining: Vec<usize> = (0..points.len())
            .filter(|i| !selected.contains(i))
            .collect();
        remaining.shuffle(rng);

        for idx in remaining {
            if selected.len() >= count {
                break;
            }
            selected.insert(idx);
        }

        selected.into_iter().collect()
    }

    fn name(&self) -> &str {
        "CentroidPlusVoxelGrid"
    }
}

/// 5. Centroid + Stratified (mini k-means then sample within)
pub struct CentroidPlusStratified {
    num_strata: usize,
    max_iterations: usize,
}

impl CentroidPlusStratified {
    pub fn new(num_strata: usize, max_iterations: usize) -> Self {
        Self {
            num_strata,
            max_iterations,
        }
    }
}

impl SamplingStrategy for CentroidPlusStratified {
    fn sample(&self, points: &[DVector<f64>], count: usize, rng: &mut impl Rng) -> Vec<usize> {
        if points.is_empty() || count == 0 {
            return vec![];
        }

        let count = count.min(points.len());
        let mut selected = HashSet::new();

        // Always include centroid
        let centroid = compute_centroid(points);
        let centroid_idx = find_closest_point(points, &centroid);
        selected.insert(centroid_idx);

        if count == 1 {
            return vec![centroid_idx];
        }

        // Mini k-means clustering
        let k = self.num_strata.min(points.len());
        let mut cluster_centers: Vec<DVector<f64>> = vec![];
        let mut indices: Vec<usize> = (0..points.len()).collect();
        indices.shuffle(rng);

        for i in 0..k {
            cluster_centers.push(points[indices[i]].clone());
        }

        let mut assignments = vec![0; points.len()];

        for _ in 0..self.max_iterations {
            // Assign points to nearest center
            for (i, point) in points.iter().enumerate() {
                let closest = cluster_centers
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let dist_a = (point - *a).norm();
                        let dist_b = (point - *b).norm();
                        dist_a.partial_cmp(&dist_b).unwrap()
                    })
                    .map(|(idx, _)| idx)
                    .unwrap();
                assignments[i] = closest;
            }

            // Update centers
            for c in 0..k {
                let cluster_point_refs: Vec<&DVector<f64>> = points
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == c)
                    .map(|(_, p)| p)
                    .collect();

                if !cluster_point_refs.is_empty() {
                    let cluster_points: Vec<DVector<f64>> =
                        cluster_point_refs.iter().map(|&p| p.clone()).collect();
                    cluster_centers[c] = compute_centroid(&cluster_points);
                }
            }
        }

        // Group indices by cluster
        let mut clusters: Vec<Vec<usize>> = vec![vec![]; k];
        for (i, &cluster_id) in assignments.iter().enumerate() {
            clusters[cluster_id].push(i);
        }

        // Sample proportionally from each stratum
        let remaining = count - 1;
        let mut per_cluster = vec![0; k];

        for i in 0..k {
            per_cluster[i] = ((clusters[i].len() as f64 / points.len() as f64) * remaining as f64)
                .round() as usize;
        }

        // Adjust to exactly match count
        let total_assigned: usize = per_cluster.iter().sum();
        if total_assigned < remaining {
            per_cluster[0] += remaining - total_assigned;
        }

        // Sample from each stratum
        for (i, cluster) in clusters.iter().enumerate() {
            let mut cluster_copy = cluster.clone();
            cluster_copy.shuffle(rng);

            for &idx in cluster_copy.iter().take(per_cluster[i]) {
                if selected.len() >= count {
                    break;
                }
                selected.insert(idx);
            }
        }

        selected.into_iter().collect()
    }

    fn name(&self) -> &str {
        "CentroidPlusStratified"
    }
}

/// 6. Pure Uniform Random (baseline without centroid requirement)
pub struct UniformRandom;

impl SamplingStrategy for UniformRandom {
    fn sample(&self, points: &[DVector<f64>], count: usize, rng: &mut impl Rng) -> Vec<usize> {
        if points.is_empty() || count == 0 {
            return vec![];
        }

        let count = count.min(points.len());
        let mut indices: Vec<usize> = (0..points.len()).collect();
        indices.shuffle(rng);
        indices.truncate(count);
        indices
    }

    fn name(&self) -> &str {
        "UniformRandom"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn make_test_points() -> Vec<DVector<f64>> {
        vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
            DVector::from_vec(vec![1.0, 1.0]),
            DVector::from_vec(vec![0.5, 0.5]),
        ]
    }

    #[test]
    fn test_centroid_uniform() {
        let points = make_test_points();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let sampler = CentroidPlusUniformRandom;
        let indices = sampler.sample(&points, 3, &mut rng);

        assert_eq!(indices.len(), 3);
        assert!(indices.contains(&4)); // Centroid is at index 4
    }

    #[test]
    fn test_fps() {
        let points = make_test_points();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let sampler = CentroidPlusFPS;
        let indices = sampler.sample(&points, 3, &mut rng);

        assert_eq!(indices.len(), 3);
        assert!(indices.contains(&4)); // Centroid included
    }

    #[test]
    fn test_voxel_grid() {
        let points = make_test_points();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let sampler = CentroidPlusVoxelGrid::new(0.3);
        let indices = sampler.sample(&points, 3, &mut rng);

        assert_eq!(indices.len(), 3);
        assert!(indices.contains(&4)); // Centroid included
    }
}
