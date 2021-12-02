use finalfusion::embeddings::Quantize;
use finalfusion::norms::NdNorms;
use finalfusion::prelude::*;
use finalfusion::storage::{QuantizedArray, Storage};
use ndarray::{CowArray, Ix1};
use pyo3::exceptions::PyValueError;
use pyo3::{PyResult, Python};
use reductive::pq::TrainPq;

pub enum EmbeddingsWrap {
    NonView(Embeddings<VocabWrap, StorageWrap>),
    View(Embeddings<VocabWrap, StorageViewWrap>),
}

impl EmbeddingsWrap {
    pub fn storage(&self) -> &dyn Storage {
        use EmbeddingsWrap::*;
        match self {
            NonView(e) => e.storage(),
            View(e) => e.storage(),
        }
    }

    pub fn vocab(&self) -> &VocabWrap {
        use EmbeddingsWrap::*;
        match self {
            NonView(e) => e.vocab(),
            View(e) => e.vocab(),
        }
    }

    pub fn norms(&self) -> Option<&NdNorms> {
        use EmbeddingsWrap::*;
        match self {
            NonView(e) => e.norms(),
            View(e) => e.norms(),
        }
    }

    pub fn embedding(&self, word: &str) -> Option<CowArray<f32, Ix1>> {
        use EmbeddingsWrap::*;
        match self {
            View(e) => e.embedding(word),
            NonView(e) => e.embedding(word),
        }
    }

    pub fn quantize<P>(
        &self,
        py: Python,
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        normalize: bool,
    ) -> PyResult<Embeddings<VocabWrap, QuantizedArray>>
    where
        P: TrainPq<f32>,
    {
        use EmbeddingsWrap::*;
        match self {
            NonView(_) => Err(PyValueError::new_err(
                "Quantization is not supported for this type of embeddings",
            )),
            View(e) => py
                .allow_threads(|| {
                    e.quantize::<P>(
                        n_subquantizers,
                        n_subquantizer_bits,
                        n_iterations,
                        n_attempts,
                        normalize,
                    )
                })
                .map_err(|err| {
                    PyValueError::new_err(format!("Error quantizing embeddings: {}", err))
                }),
        }
    }

    pub fn view(&self) -> Option<&Embeddings<VocabWrap, StorageViewWrap>> {
        match self {
            EmbeddingsWrap::NonView(_) => None,
            EmbeddingsWrap::View(storage) => Some(storage),
        }
    }
}
