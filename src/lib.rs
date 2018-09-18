#![feature(specialization)]

extern crate failure;
extern crate finalfrontier;
extern crate ndarray;
extern crate pyo3;

use std::fs::File;
use std::io::BufReader;
use std::rc::Rc;

use failure::Error;
use finalfrontier::similarity::{Analogy, Similarity};
use finalfrontier::{MmapModelBinary, Model, ReadModelBinary};
use ndarray::Axis;
use pyo3::prelude::*;

/// This is a binding for finalfrontier.
///
/// finalfrontier is a library and set of programs for training
/// word embeddings with subword units. The Python binding can
/// be used to query the resulting embeddings and do similarity
/// queries.
#[pymodinit]
fn finalfrontier(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyWordSimilarity>()?;
    Ok(())
}

/// A word and its similarity to a query word.
///
/// The similarity is normally a value between -1 (opposite
/// vectors) and 1 (identical vectors).
#[pyclass(name=WordSimilarity)]
struct PyWordSimilarity {
    #[prop(get)]
    word: String,

    #[prop(get)]
    similarity: f32,

    token: PyToken,
}

#[pyproto]
impl PyObjectProtocol for PyWordSimilarity {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "WordSimilarity('{}', {})",
            self.word, self.similarity
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}: {}", self.word, self.similarity))
    }
}

/// A finalfrontier model.
#[pyclass(name=Model)]
struct PyModel {
    model: Rc<Model>,
    token: PyToken,
}

#[pymethods]
impl PyModel {
    /// Load a model from the given `path`.
    ///
    /// When the `mmap` argument is `True`, the embedding matrix is
    /// not loaded into memory, but memory mapped. This results in
    /// lower memory use and shorter model load times, while sacrificing
    /// some query efficiency.
    #[new]
    #[args(mmap = false)]
    fn __new__(obj: &PyRawObject, path: &str, mmap: bool) -> PyResult<()> {
        let model = match load_model(path, mmap) {
            Ok(model) => Rc::new(model),
            Err(err) => {
                return Err(exc::IOError::py_err(err.to_string()));
            }
        };

        obj.init(|token| PyModel { model, token })
    }

    /// Perform an anology query.
    ///
    /// This returns words for the analogy query *w1* is to *w2*
    /// as *w3* is to ?.
    #[args(limit = 10)]
    fn analogy(
        &self,
        py: Python,
        word1: &str,
        word2: &str,
        word3: &str,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let results = match self.model.analogy(word1, word2, word3, limit) {
            Some(results) => results,
            None => return Err(exc::KeyError::py_err("Unknown word and n-grams")),
        };

        let mut r = Vec::with_capacity(results.len());
        for ws in results {
            r.push(
                Py::new(py, |token| PyWordSimilarity {
                    word: ws.word.to_owned(),
                    similarity: ws.similarity.into_inner(),
                    token,
                })?.into_object(py),
            )
        }

        Ok(r)
    }

    /// Get the embedding for the given word.
    ///
    /// If the word is not known, its representation is approximated
    /// using subword units.
    fn embedding(&self, word: &str) -> PyResult<Vec<f32>> {
        match self.model.embedding(word) {
            Some(embedding) => Ok(embedding.to_vec()),
            None => Err(exc::KeyError::py_err("Unknown word and n-grams")),
        }
    }

    /// Perform a similarity query.
    #[args(limit = 10)]
    fn similarity(&self, py: Python, word: &str, limit: usize) -> PyResult<Vec<PyObject>> {
        let results = match self.model.similarity(word, limit) {
            Some(results) => results,
            None => return Err(exc::KeyError::py_err("Unknown word and n-grams")),
        };

        let mut r = Vec::with_capacity(results.len());
        for ws in results {
            r.push(
                Py::new(py, |token| PyWordSimilarity {
                    word: ws.word.to_owned(),
                    similarity: ws.similarity.into_inner(),
                    token,
                })?.into_object(py),
            )
        }

        Ok(r)
    }
}

#[pyproto]
impl PyIterProtocol for PyModel {
    fn __iter__(&mut self) -> PyResult<PyObject> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let iter = Py::new(py, |token| PyModelIterator {
            model: self.model.clone(),
            idx: 0,
            token,
        })?.into_object(py);

        Ok(iter)
    }
}

fn load_model(path: &str, mmap: bool) -> Result<Model, Error> {
    let f = File::open(path)?;

    let model = if mmap {
        Model::mmap_model_binary(f)?
    } else {
        Model::read_model_binary(&mut BufReader::new(f))?
    };

    Ok(model)
}

#[pyclass(name=ModelIterator)]
struct PyModelIterator {
    model: Rc<Model>,
    idx: usize,
    token: PyToken,
}

#[pyproto]
impl PyIterProtocol for PyModelIterator {
    fn __iter__(&mut self) -> PyResult<PyObject> {
        Ok(self.into())
    }

    fn __next__(&mut self) -> PyResult<Option<(String, Vec<f32>)>> {
        let vocab = self.model.vocab();
        let embeddings = self.model.embedding_matrix();

        if self.idx < vocab.len() {
            let word = vocab.words()[self.idx].word().to_string();
            let embed = embeddings.subview(Axis(0), self.idx).to_vec();
            self.idx += 1;
            Ok(Some((word, embed)))
        } else {
            Ok(None)
        }
    }
}