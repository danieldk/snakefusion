use pyo3::prelude::*;

#[pyclass(name = "Config")]
pub struct PyConfig;

#[pymethods]
impl PyConfig {
    #[classattr]
    pub const OPQ: bool = cfg!(feature = "opq");
}
