use nalgebra::DVector;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod utils;

use utils::sample_measures;
use utils::size_measures;

/// Convert Python list of lists to Vec<DVector<f64>>
fn parse_points(points: Vec<Vec<f64>>) -> PyResult<Vec<DVector<f64>>> {
    if points.is_empty() {
        return Ok(vec![]);
    }
    let dim = points[0].len();
    if dim == 0 {
        return Err(PyValueError::new_err(
            "Points must have at least one dimension",
        ));
    }
    let mut result = Vec::with_capacity(points.len());
    for (i, point) in points.iter().enumerate() {
        if point.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Point {} has dimension {}, expected {}",
                i,
                point.len(),
                dim
            )));
        }
        result.push(DVector::from_vec(point.clone()));
    }
    Ok(result)
}

/// Convert numpy array (n_points × n_dims) to Vec<DVector<f64>>
fn parse_numpy_points(arr: PyReadonlyArray2<f64>) -> PyResult<Vec<DVector<f64>>> {
    let array = arr.as_array();
    let (n_points, n_dims) = array.dim();
    if n_dims == 0 {
        return Err(PyValueError::new_err(
            "Points must have at least one dimension",
        ));
    }
    let result: Vec<DVector<f64>> = (0..n_points)
        .map(|i| DVector::from_vec(array.row(i).to_vec()))
        .collect();
    Ok(result)
}

// ============ Size Measures - List-based API ============

#[pyfunction]
fn bbox_volume(points: Vec<Vec<f64>>) -> PyResult<f64> {
    let parsed = parse_points(points)?;
    Ok(size_measures::size_bbox(&parsed))
}

#[pyfunction]
fn hull_volume(points: Vec<Vec<f64>>) -> PyResult<f64> {
    let parsed = parse_points(points)?;
    Ok(size_measures::size_hull(&parsed))
}

#[pyfunction]
fn covariance_volume(points: Vec<Vec<f64>>) -> PyResult<f64> {
    let parsed = parse_points(points)?;
    Ok(size_measures::size_covdet(&parsed))
}

#[pyfunction]
fn mean_dispersion(points: Vec<Vec<f64>>) -> PyResult<f64> {
    let parsed = parse_points(points)?;
    Ok(size_measures::size_dispersion(&parsed))
}

// ============ Size Measures - NumPy API ============

#[pyfunction]
fn bbox_volume_numpy(points: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let parsed = parse_numpy_points(points)?;
    Ok(size_measures::size_bbox(&parsed))
}

#[pyfunction]
fn hull_volume_numpy(points: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let parsed = parse_numpy_points(points)?;
    Ok(size_measures::size_hull(&parsed))
}

#[pyfunction]
fn covariance_volume_numpy(points: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let parsed = parse_numpy_points(points)?;
    Ok(size_measures::size_covdet(&parsed))
}

#[pyfunction]
fn mean_dispersion_numpy(points: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let parsed = parse_numpy_points(points)?;
    Ok(size_measures::size_dispersion(&parsed))
}

// ============ Sampling - List-based API ============

#[pyfunction]
fn sample_centroid_uniform(points: Vec<Vec<f64>>, count: usize, seed: u64) -> PyResult<Vec<usize>> {
    let parsed = parse_points(points)?;
    Ok(sample_measures::sample_centroid_uniform(
        &parsed, count, seed,
    ))
}

#[pyfunction]
fn sample_centroid_fps(points: Vec<Vec<f64>>, count: usize, seed: u64) -> PyResult<Vec<usize>> {
    let parsed = parse_points(points)?;
    Ok(sample_measures::sample_centroid_fps(&parsed, count, seed))
}

#[pyfunction]
fn sample_centroid_kmedoids(
    points: Vec<Vec<f64>>,
    count: usize,
    seed: u64,
    max_iterations: usize,
) -> PyResult<Vec<usize>> {
    let parsed = parse_points(points)?;
    Ok(sample_measures::sample_centroid_kmedoids(
        &parsed,
        count,
        seed,
        max_iterations,
    ))
}

#[pyfunction]
fn sample_centroid_voxel(
    points: Vec<Vec<f64>>,
    count: usize,
    seed: u64,
    voxel_size: f64,
) -> PyResult<Vec<usize>> {
    let parsed = parse_points(points)?;
    Ok(sample_measures::sample_centroid_voxel(
        &parsed, count, seed, voxel_size,
    ))
}

#[pyfunction]
fn sample_centroid_stratified(
    points: Vec<Vec<f64>>,
    count: usize,
    seed: u64,
    num_strata: usize,
    max_iterations: usize,
) -> PyResult<Vec<usize>> {
    let parsed = parse_points(points)?;
    Ok(sample_measures::sample_centroid_stratified(
        &parsed,
        count,
        seed,
        num_strata,
        max_iterations,
    ))
}

#[pyfunction]
fn sample_uniform(points: Vec<Vec<f64>>, count: usize, seed: u64) -> PyResult<Vec<usize>> {
    let parsed = parse_points(points)?;
    Ok(sample_measures::sample_uniform(&parsed, count, seed))
}

// ============ Sampling - NumPy API ============

#[pyfunction]
fn sample_centroid_uniform_numpy(
    points: PyReadonlyArray2<f64>,
    count: usize,
    seed: u64,
) -> PyResult<Vec<usize>> {
    let parsed = parse_numpy_points(points)?;
    Ok(sample_measures::sample_centroid_uniform(
        &parsed, count, seed,
    ))
}

#[pyfunction]
fn sample_centroid_fps_numpy(
    points: PyReadonlyArray2<f64>,
    count: usize,
    seed: u64,
) -> PyResult<Vec<usize>> {
    let parsed = parse_numpy_points(points)?;
    Ok(sample_measures::sample_centroid_fps(&parsed, count, seed))
}

#[pyfunction]
fn sample_centroid_kmedoids_numpy(
    points: PyReadonlyArray2<f64>,
    count: usize,
    seed: u64,
    max_iterations: usize,
) -> PyResult<Vec<usize>> {
    let parsed = parse_numpy_points(points)?;
    Ok(sample_measures::sample_centroid_kmedoids(
        &parsed,
        count,
        seed,
        max_iterations,
    ))
}

#[pyfunction]
fn sample_centroid_voxel_numpy(
    points: PyReadonlyArray2<f64>,
    count: usize,
    seed: u64,
    voxel_size: f64,
) -> PyResult<Vec<usize>> {
    let parsed = parse_numpy_points(points)?;
    Ok(sample_measures::sample_centroid_voxel(
        &parsed, count, seed, voxel_size,
    ))
}

#[pyfunction]
fn sample_centroid_stratified_numpy(
    points: PyReadonlyArray2<f64>,
    count: usize,
    seed: u64,
    num_strata: usize,
    max_iterations: usize,
) -> PyResult<Vec<usize>> {
    let parsed = parse_numpy_points(points)?;
    Ok(sample_measures::sample_centroid_stratified(
        &parsed,
        count,
        seed,
        num_strata,
        max_iterations,
    ))
}

#[pyfunction]
fn sample_uniform_numpy(
    points: PyReadonlyArray2<f64>,
    count: usize,
    seed: u64,
) -> PyResult<Vec<usize>> {
    let parsed = parse_numpy_points(points)?;
    Ok(sample_measures::sample_uniform(&parsed, count, seed))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Size Measures - List-based API
    m.add_function(wrap_pyfunction!(bbox_volume, m)?)?;
    m.add_function(wrap_pyfunction!(hull_volume, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_volume, m)?)?;
    m.add_function(wrap_pyfunction!(mean_dispersion, m)?)?;

    // Size Measures - NumPy API
    m.add_function(wrap_pyfunction!(bbox_volume_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(hull_volume_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_volume_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(mean_dispersion_numpy, m)?)?;

    // Sampling - List-based API
    m.add_function(wrap_pyfunction!(sample_centroid_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(sample_centroid_fps, m)?)?;
    m.add_function(wrap_pyfunction!(sample_centroid_kmedoids, m)?)?;
    m.add_function(wrap_pyfunction!(sample_centroid_voxel, m)?)?;
    m.add_function(wrap_pyfunction!(sample_centroid_stratified, m)?)?;
    m.add_function(wrap_pyfunction!(sample_uniform, m)?)?;

    // Sampling - NumPy API
    m.add_function(wrap_pyfunction!(sample_centroid_uniform_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(sample_centroid_fps_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(sample_centroid_kmedoids_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(sample_centroid_voxel_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(sample_centroid_stratified_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(sample_uniform_numpy, m)?)?;

    Ok(())
}
