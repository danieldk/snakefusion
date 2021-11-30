use std::sync::{Arc, RwLock};

use finalfusion::vocab::{NGramIndices, SubwordIndices, Vocab, VocabWrap, WordIndex};
use pyo3::class::sequence::PySequenceProtocol;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::{PyIterProtocol, PyMappingProtocol};

use crate::iter::PyVocabIterator;
use crate::EmbeddingsWrap;

type NGramIndex = (String, Vec<usize>);

/// finalfusion vocab.
#[pyclass(name = "Vocab")]
pub struct PyVocab {
    embeddings: Arc<RwLock<EmbeddingsWrap>>,
}

impl PyVocab {
    pub fn new(embeddings: Arc<RwLock<EmbeddingsWrap>>) -> Self {
        PyVocab { embeddings }
    }
}

#[pymethods]
impl PyVocab {
    #[args(default = "Python::acquire_gil().python().None()")]
    fn get(&self, key: &str, default: PyObject) -> Option<PyObject> {
        let embeds = self.embeddings.read().unwrap();
        let gil = pyo3::Python::acquire_gil();
        let idx = embeds.vocab().idx(key).map(|idx| match idx {
            WordIndex::Word(idx) => idx.to_object(gil.python()),
            WordIndex::Subword(indices) => indices.to_object(gil.python()),
        });
        if !default.is_none(gil.python()) && idx.is_none() {
            return Some(default);
        }
        idx
    }

    fn ngram_indices(&self, word: &str) -> PyResult<Option<Vec<NGramIndex>>> {
        let embeds = self.embeddings.read().unwrap();
        let indices = match embeds.vocab() {
            VocabWrap::FastTextSubwordVocab(inner) => inner.ngram_indices(word),
            VocabWrap::BucketSubwordVocab(inner) => inner.ngram_indices(word),
            VocabWrap::ExplicitSubwordVocab(inner) => inner.ngram_indices(word),
            VocabWrap::FloretSubwordVocab(inner) => inner.ngram_indices(word),
            VocabWrap::SimpleVocab(_) => {
                return Err(PyValueError::new_err(
                    "querying n-gram indices is not supported for this vocabulary",
                ))
            }
        };

        let indices = indices.map(|indices| {
            indices
                .into_iter()
                .map(|idx| (idx.0, idx.1.into_vec()))
                .collect()
        });

        Ok(indices)
    }

    fn subword_indices(&self, word: &str) -> PyResult<Option<Vec<usize>>> {
        let embeds = self.embeddings.read().unwrap();
        match embeds.vocab() {
            VocabWrap::FastTextSubwordVocab(inner) => Ok(inner.subword_indices(word)),
            VocabWrap::BucketSubwordVocab(inner) => Ok(inner.subword_indices(word)),
            VocabWrap::ExplicitSubwordVocab(inner) => Ok(inner.subword_indices(word)),
            VocabWrap::FloretSubwordVocab(inner) => Ok(inner.subword_indices(word)),
            VocabWrap::SimpleVocab(_) => Err(PyValueError::new_err(
                "querying subwords' indices is not supported for this vocabulary",
            )),
        }
    }
}

impl PyVocab {
    fn str_to_indices(&self, query: &str) -> PyResult<WordIndex> {
        let embeds = self.embeddings.read().unwrap();
        embeds
            .vocab()
            .idx(query)
            .ok_or_else(|| PyKeyError::new_err(format!("key not found: '{}'", query)))
    }

    fn validate_and_convert_isize_idx(&self, mut idx: isize) -> PyResult<usize> {
        let embeds = self.embeddings.read().unwrap();
        let vocab = embeds.vocab();
        if idx < 0 {
            idx += vocab.words_len() as isize;
        }

        if idx >= vocab.words_len() as isize || idx < 0 {
            Err(PyIndexError::new_err("list index out of range"))
        } else {
            Ok(idx as usize)
        }
    }
}

#[pyproto]
impl PyMappingProtocol for PyVocab {
    fn __getitem__(&self, query: PyObject) -> PyResult<PyObject> {
        let embeds = self.embeddings.read().unwrap();
        let vocab = embeds.vocab();
        let gil = Python::acquire_gil();
        if let Ok(idx) = query.extract::<isize>(gil.python()) {
            let idx = self.validate_and_convert_isize_idx(idx)?;
            return Ok(vocab.words()[idx].clone().into_py(gil.python()));
        }

        if let Ok(query) = query.extract::<String>(gil.python()) {
            return self.str_to_indices(&query).map(|idx| match idx {
                WordIndex::Subword(indices) => indices.into_py(gil.python()),
                WordIndex::Word(idx) => idx.into_py(gil.python()),
            });
        }

        Err(PyKeyError::new_err("key must be integer or string"))
    }
}

#[pyproto]
impl PyIterProtocol for PyVocab {
    fn __iter__(slf: PyRefMut<Self>) -> PyResult<PyVocabIterator> {
        Ok(PyVocabIterator::new(slf.embeddings.clone(), 0))
    }
}

#[pyproto]
impl PySequenceProtocol for PyVocab {
    fn __len__(&self) -> PyResult<usize> {
        let embeds = self.embeddings.read().unwrap();
        Ok(embeds.vocab().words_len())
    }

    fn __contains__(&self, word: String) -> PyResult<bool> {
        let embeds = self.embeddings.read().unwrap();
        Ok(embeds
            .vocab()
            .idx(&word)
            .and_then(|word_idx| word_idx.word())
            .is_some())
    }
}
