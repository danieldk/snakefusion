#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "intel-mkl")]
extern crate intel_mkl_src;

use pyo3::prelude::*;

mod config;
use config::PyConfig;

mod embeddings;
use embeddings::PyEmbeddings;

mod embeddings_wrap;
use embeddings_wrap::EmbeddingsWrap;

mod iter;
use iter::{PyEmbedding, PyEmbeddingIterator};

mod pq;
use pq::PyQuantizer;

mod similarity;
use similarity::PyWordSimilarity;

mod vocab;
use vocab::PyVocab;

mod storage;
use storage::PyStorage;

/// This is a Python module for using finalfusion embeddings.
///
/// finalfusion is a format for word embeddings that supports words,
/// subwords, memory-mapped matrices, and quantized matrices.
#[pymodule]
fn snakefusion(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<PyConfig>()?;
    m.add_class::<PyEmbeddings>()?;
    m.add_class::<PyEmbedding>()?;
    m.add_class::<PyQuantizer>()?;
    m.add_class::<PyStorage>()?;
    m.add_class::<PyWordSimilarity>()?;
    m.add_class::<PyVocab>()?;
    Ok(())
}
