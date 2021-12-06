use std::sync::{Arc, RwLock};

use finalfusion::vocab::{NGramIndices, SubwordIndices, Vocab, VocabWrap, WordIndex};
use pyo3::class::sequence::PySequenceProtocol;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::PyIterProtocol;

use crate::iter::PyVocabIterator;
use crate::EmbeddingsWrap;

type NGramIndex = (String, Vec<usize>);

/// Embeddings vocabulary
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
    fn __getitem__(&self, py: Python, query: PyObject) -> PyResult<PyObject> {
        let embeds = self.embeddings.read().unwrap();
        let vocab = embeds.vocab();
        if let Ok(idx) = query.extract::<isize>(py) {
            let idx = self.validate_and_convert_isize_idx(idx)?;
            return Ok(vocab.words()[idx].clone().into_py(py));
        }

        if let Ok(query) = query.extract::<String>(py) {
            return self.str_to_indices(py, &query).map(|idx| match idx {
                WordIndex::Subword(indices) => indices.into_py(py),
                WordIndex::Word(idx) => idx.into_py(py),
            });
        }

        Err(PyKeyError::new_err("key must be integer or string"))
    }

    /// get(self, word, /, default=None)
    /// --
    ///
    /// Get the index or subword indices of a word.
    ///
    /// If a word is known, returns the index of the word in the
    /// embedding matrix. If a word is unknown, return its subword
    /// indices.
    ///
    /// The provided `default` parameter is returned if the word
    /// could not be looked up.
    #[args(default = "Python::acquire_gil().python().None()")]
    fn get(&self, py: Python, key: &str, default: PyObject) -> Option<PyObject> {
        let embeds = self.embeddings.read().unwrap();
        let idx = py
            .allow_threads(|| embeds.vocab().idx(key))
            .map(|idx| match idx {
                WordIndex::Word(idx) => idx.to_object(py),
                WordIndex::Subword(indices) => indices.to_object(py),
            });
        if !default.is_none(py) && idx.is_none() {
            return Some(default);
        }
        idx
    }

    /// ngram_indices(self, word)
    /// --
    ///
    /// Return the of a word and their indices.
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

    /// subword_indices(self, word)
    /// --
    ///
    /// Return the subword indices of a word.
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
    fn str_to_indices(&self, py: Python, query: &str) -> PyResult<WordIndex> {
        let embeds = self.embeddings.read().unwrap();
        py.allow_threads(|| embeds.vocab().idx(query))
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
