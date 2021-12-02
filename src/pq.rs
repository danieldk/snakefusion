use pyo3::prelude::*;

#[pyclass(name = "Quantizer")]
pub struct PyQuantizer;

#[pymethods]
impl PyQuantizer {
    #[classattr]
    const PQ: usize = 0;

    #[classattr]
    const OPQ: usize = 1;

    #[classattr]
    const GAUSSIAN_OPQ: usize = 2;
}
