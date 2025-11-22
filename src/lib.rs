use nalgebra::DVector;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod utils;
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

// ============ List-based API ============

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

// ============ NumPy API ============

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

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // List-based API
    m.add_function(wrap_pyfunction!(bbox_volume, m)?)?;
    m.add_function(wrap_pyfunction!(hull_volume, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_volume, m)?)?;
    m.add_function(wrap_pyfunction!(mean_dispersion, m)?)?;

    // NumPy API
    m.add_function(wrap_pyfunction!(bbox_volume_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(hull_volume_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_volume_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(mean_dispersion_numpy, m)?)?;

    Ok(())
}
