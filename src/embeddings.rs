use std::cell::RefCell;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::rc::Rc;

use finalfusion::chunks::metadata::Metadata;
use finalfusion::compat::text::{ReadText, ReadTextDims};
use finalfusion::compat::word2vec::ReadWord2Vec;
use finalfusion::io as ffio;
use finalfusion::prelude::*;
use finalfusion::similarity::*;
use itertools::Itertools;
use ndarray::Array1;
use numpy::{IntoPyArray, NpyDataType, PyArray1};
use pyo3::class::iter::PyIterProtocol;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyTuple};
use pyo3::{exceptions, PyMappingProtocol};
use toml::{self, Value};

use crate::storage::PyStorage;
use crate::{EmbeddingsWrap, PyEmbeddingIterator, PyVocab, PyWordSimilarity};

/// finalfusion embeddings.
#[pyclass(name = Embeddings)]
pub struct PyEmbeddings {
    // The use of Rc + RefCell should be safe in this crate:
    //
    // 1. Python is single-threaded.
    // 2. The only mutable borrow (in set_metadata) is limited
    //    to its method scope.
    // 3. None of the methods returns borrowed embeddings.
    embeddings: Rc<RefCell<EmbeddingsWrap>>,
}

#[pymethods]
impl PyEmbeddings {
    /// Load embeddings from the given `path`.
    ///
    /// When the `mmap` argument is `True`, the embedding matrix is
    /// not loaded into memory, but memory mapped. This results in
    /// lower memory use and shorter load times, while sacrificing
    /// some query efficiency.
    #[new]
    #[args(mmap = false)]
    fn __new__(obj: &PyRawObject, path: &str, mmap: bool) -> PyResult<()> {
        // First try to load embeddings with viewable storage. If that
        // fails, attempt to load the embeddings as non-viewable
        // storage.
        let embeddings = match read_embeddings(path, mmap) {
            Ok(e) => Rc::new(RefCell::new(EmbeddingsWrap::View(e))),
            Err(_) => read_embeddings(path, mmap)
                .map(|e| Rc::new(RefCell::new(EmbeddingsWrap::NonView(e))))
                .map_err(|err| exceptions::IOError::py_err(err.to_string()))?,
        };

        obj.init(PyEmbeddings { embeddings });

        Ok(())
    }

    /// read_fasttext(path,/ lossy)
    /// --
    ///
    /// Read embeddings in the fasttext format.
    ///
    /// Lossy decoding of the words can be toggled through the lossy param.
    #[staticmethod]
    #[args(lossy = false)]
    fn read_fasttext(path: &str, lossy: bool) -> PyResult<PyEmbeddings> {
        if lossy {
            read_non_fifu_embeddings(path, |r| Embeddings::read_fasttext_lossy(r))
        } else {
            read_non_fifu_embeddings(path, |r| Embeddings::read_fasttext(r))
        }
    }

    /// read_text(path,/ lossy)
    /// --
    ///
    /// Read embeddings in text format. This format uses one line per
    /// embedding. Each line starts with the word in UTF-8, followed
    /// by its vector components encoded in ASCII. The word and its
    /// components are separated by spaces.
    ///
    /// Lossy decoding of the words can be toggled through the lossy param.
    #[staticmethod]
    #[args(lossy = false)]
    fn read_text(path: &str, lossy: bool) -> PyResult<PyEmbeddings> {
        if lossy {
            read_non_fifu_embeddings(path, |r| Embeddings::read_text_lossy(r))
        } else {
            read_non_fifu_embeddings(path, |r| Embeddings::read_text(r))
        }
    }

    /// read_text_dims(path,/ lossy)
    /// --
    ///
    /// Read embeddings in text format with dimensions. In this format,
    /// the first line states the shape of the embedding matrix. The
    /// number of rows (words) and columns (embedding dimensionality) is
    /// separated by a space character. The remainder of the file uses
    /// one line per embedding. Each line starts with the word in UTF-8,
    /// followed by its vector components encoded in ASCII. The word and
    /// its components are separated by spaces.
    ///
    /// Lossy decoding of the words can be toggled through the lossy param.
    #[staticmethod]
    #[args(lossy = false)]
    fn read_text_dims(path: &str, lossy: bool) -> PyResult<PyEmbeddings> {
        if lossy {
            read_non_fifu_embeddings(path, |r| Embeddings::read_text_dims_lossy(r))
        } else {
            read_non_fifu_embeddings(path, |r| Embeddings::read_text_dims(r))
        }
    }

    /// read_word2vec(path,/ lossy)
    /// --
    ///
    /// Read embeddings in the word2vec binary format.
    ///
    /// Lossy decoding of the words can be toggled through the lossy param.
    #[staticmethod]
    #[args(lossy = false)]
    fn read_word2vec(path: &str, lossy: bool) -> PyResult<PyEmbeddings> {
        if lossy {
            read_non_fifu_embeddings(path, |r| Embeddings::read_word2vec_binary_lossy(r))
        } else {
            read_non_fifu_embeddings(path, |r| Embeddings::read_word2vec_binary(r))
        }
    }

    /// Get the model's vocabulary.
    fn vocab(&self) -> PyResult<PyVocab> {
        Ok(PyVocab::new(self.embeddings.clone()))
    }

    /// Get the model's storage.
    fn storage(&self) -> PyStorage {
        PyStorage::new(self.embeddings.clone())
    }

    /// Perform an anology query.
    ///
    /// This returns words for the analogy query *w1* is to *w2*
    /// as *w3* is to ?.
    #[args(limit = 10, mask = "(true, true, true)")]
    fn analogy(
        &self,
        py: Python,
        word1: &str,
        word2: &str,
        word3: &str,
        limit: usize,
        mask: (bool, bool, bool),
    ) -> PyResult<Vec<PyObject>> {
        let embeddings = self.embeddings.borrow();

        let embeddings = embeddings.view().ok_or_else(|| {
            exceptions::ValueError::py_err(
                "Analogy queries are not supported for this type of embedding matrix",
            )
        })?;

        let results = embeddings
            .analogy_masked([word1, word2, word3], [mask.0, mask.1, mask.2], limit)
            .map_err(|lookup| {
                let failed = [word1, word2, word3]
                    .iter()
                    .zip(lookup.iter())
                    .filter(|(_, success)| !*success)
                    .map(|(word, _)| word)
                    .join(" ");
                exceptions::KeyError::py_err(format!("Unknown word or n-grams: {}", failed))
            })?;

        Self::similarity_results(py, results)
    }

    /// embedding(word,/, default)
    /// --
    ///
    /// Get the embedding for the given word.
    ///
    /// If the word is not known, its representation is approximated
    /// using subword units. #
    ///
    /// If no representation can be calculated:
    ///  - `None` if `default` is `None`
    ///  - an array filled with `default` if `default` is a scalar
    ///  - an array if `default` is a 1-d array
    ///  - an array filled with values from `default` if it is an iterator over floats.
    #[args(default = "PyEmbeddingDefault::default()")]
    fn embedding(
        &self,
        word: &str,
        default: PyEmbeddingDefault,
    ) -> PyResult<Option<Py<PyArray1<f32>>>> {
        let embeddings = self.embeddings.borrow();
        let gil = pyo3::Python::acquire_gil();
        if let PyEmbeddingDefault::Embedding(array) = &default {
            if array.as_ref(gil.python()).shape()[0] != embeddings.storage().shape().1 {
                return Err(exceptions::ValueError::py_err(format!(
                    "Invalid shape of default embedding: {}",
                    array.as_ref(gil.python()).shape()[0]
                )));
            }
        }

        if let Some(embedding) = embeddings.embedding(word) {
            return Ok(Some(
                embedding.into_owned().into_pyarray(gil.python()).to_owned(),
            ));
        };
        match default {
            PyEmbeddingDefault::Constant(constant) => {
                let nd_arr = Array1::from_elem([embeddings.storage().shape().1], constant);
                Ok(Some(nd_arr.into_pyarray(gil.python()).to_owned()))
            }
            PyEmbeddingDefault::Embedding(array) => Ok(Some(array)),
            PyEmbeddingDefault::None => Ok(None),
        }
    }

    fn embedding_with_norm(&self, word: &str) -> Option<Py<PyTuple>> {
        let embeddings = self.embeddings.borrow();

        use EmbeddingsWrap::*;
        let embedding_with_norm = match &*embeddings {
            View(e) => e.embedding_with_norm(word),
            NonView(e) => e.embedding_with_norm(word),
        };

        embedding_with_norm.map(|e| {
            let gil = pyo3::Python::acquire_gil();
            let embedding = e.embedding.into_owned().into_pyarray(gil.python());
            (embedding, e.norm).into_py(gil.python())
        })
    }

    /// Embeddings metadata.
    #[getter]
    fn metadata(&self) -> PyResult<Option<String>> {
        let embeddings = self.embeddings.borrow();

        use EmbeddingsWrap::*;
        let metadata = match &*embeddings {
            View(e) => e.metadata(),
            NonView(e) => e.metadata(),
        };

        match metadata.map(|v| toml::ser::to_string_pretty(&v.0)) {
            Some(Ok(toml)) => Ok(Some(toml)),
            Some(Err(err)) => Err(exceptions::IOError::py_err(format!(
                "Metadata is invalid TOML: {}",
                err
            ))),
            None => Ok(None),
        }
    }

    #[setter]
    fn set_metadata(&mut self, metadata: &str) -> PyResult<()> {
        let value = match metadata.parse::<Value>() {
            Ok(value) => value,
            Err(err) => {
                return Err(exceptions::ValueError::py_err(format!(
                    "Metadata is invalid TOML: {}",
                    err
                )));
            }
        };

        let mut embeddings = self.embeddings.borrow_mut();

        use EmbeddingsWrap::*;
        match &mut *embeddings {
            View(e) => e.set_metadata(Some(Metadata(value))),
            NonView(e) => e.set_metadata(Some(Metadata(value))),
        };

        Ok(())
    }

    /// Perform a similarity query.
    #[args(limit = 10)]
    fn word_similarity(&self, py: Python, word: &str, limit: usize) -> PyResult<Vec<PyObject>> {
        let embeddings = self.embeddings.borrow();

        let embeddings = embeddings.view().ok_or_else(|| {
            exceptions::ValueError::py_err(
                "Similarity queries are not supported for this type of embedding matrix",
            )
        })?;

        let results = embeddings
            .word_similarity(word, limit)
            .ok_or_else(|| exceptions::KeyError::py_err("Unknown word and n-grams"))?;

        Self::similarity_results(py, results)
    }

    /// Perform a similarity query based on a query embedding.
    #[args(limit = 10, skip = "Skips(HashSet::new())")]
    fn embedding_similarity(
        &self,
        py: Python,
        embedding: PyEmbedding,
        skip: Skips,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let embeddings = self.embeddings.borrow();

        let embeddings = embeddings.view().ok_or_else(|| {
            exceptions::ValueError::py_err(
                "Similarity queries are not supported for this type of embedding matrix",
            )
        })?;

        let embedding = embedding.0.as_array();

        if embedding.shape()[0] != embeddings.storage().shape().1 {
            return Err(exceptions::ValueError::py_err(format!(
                "Incompatible embedding shapes: embeddings: ({},), query: ({},)",
                embedding.shape()[0],
                embeddings.storage().shape().1
            )));
        }

        let results = embeddings.embedding_similarity_masked(embedding, limit, &skip.0);

        Self::similarity_results(
            py,
            results.ok_or_else(|| exceptions::KeyError::py_err("Unknown word and n-grams"))?,
        )
    }

    /// Write the embeddings to a finalfusion file.
    fn write(&self, filename: &str) -> PyResult<()> {
        let f = File::create(filename)?;
        let mut writer = BufWriter::new(f);

        let embeddings = self.embeddings.borrow();

        use EmbeddingsWrap::*;
        match &*embeddings {
            View(e) => e
                .write_embeddings(&mut writer)
                .map_err(|err| exceptions::IOError::py_err(err.to_string())),
            NonView(e) => e
                .write_embeddings(&mut writer)
                .map_err(|err| exceptions::IOError::py_err(err.to_string())),
        }
    }
}

impl PyEmbeddings {
    fn similarity_results(
        py: Python,
        results: Vec<WordSimilarityResult>,
    ) -> PyResult<Vec<PyObject>> {
        let mut r = Vec::with_capacity(results.len());
        for ws in results {
            r.push(IntoPy::into_py(
                Py::new(
                    py,
                    PyWordSimilarity::new(ws.word.to_owned(), ws.similarity.into_inner()),
                )?,
                py,
            ))
        }
        Ok(r)
    }
}

#[pyproto]
impl PyMappingProtocol for PyEmbeddings {
    fn __getitem__(&self, word: &str) -> PyResult<Py<PyArray1<f32>>> {
        let embeddings = self.embeddings.borrow();

        match embeddings.embedding(word) {
            Some(embedding) => {
                let gil = pyo3::Python::acquire_gil();
                Ok(embedding.into_owned().into_pyarray(gil.python()).to_owned())
            }
            None => Err(exceptions::KeyError::py_err("Unknown word and n-grams")),
        }
    }
}

#[pyproto]
impl PyIterProtocol for PyEmbeddings {
    fn __iter__(slf: PyRefMut<Self>) -> PyResult<PyObject> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let iter = IntoPy::into_py(
            Py::new(py, PyEmbeddingIterator::new(slf.embeddings.clone(), 0))?,
            py,
        );

        Ok(iter)
    }
}

fn read_embeddings<S>(path: &str, mmap: bool) -> Result<Embeddings<VocabWrap, S>, ffio::Error>
where
    Embeddings<VocabWrap, S>: ReadEmbeddings + MmapEmbeddings,
{
    let f = File::open(path)
        .map_err(|e| ffio::ErrorKind::io_error("Cannot open embeddings file for reading", e))?;
    let mut reader = BufReader::new(f);

    let embeddings = if mmap {
        Embeddings::mmap_embeddings(&mut reader)?
    } else {
        Embeddings::read_embeddings(&mut reader)?
    };

    Ok(embeddings)
}

fn read_non_fifu_embeddings<R, V>(path: &str, read_embeddings: R) -> PyResult<PyEmbeddings>
where
    R: FnOnce(&mut BufReader<File>) -> ffio::Result<Embeddings<V, NdArray>>,
    V: Vocab,
    Embeddings<VocabWrap, StorageViewWrap>: From<Embeddings<V, NdArray>>,
{
    let f = File::open(path).map_err(|err| {
        exceptions::IOError::py_err(format!(
            "Cannot read text embeddings from '{}': {}'",
            path, err
        ))
    })?;
    let mut reader = BufReader::new(f);

    let embeddings = read_embeddings(&mut reader).map_err(|err| {
        exceptions::IOError::py_err(format!(
            "Cannot read text embeddings from '{}': {}'",
            path, err
        ))
    })?;

    Ok(PyEmbeddings {
        embeddings: Rc::new(RefCell::new(EmbeddingsWrap::View(embeddings.into()))),
    })
}

pub enum PyEmbeddingDefault {
    Embedding(Py<PyArray1<f32>>),
    Constant(f32),
    None,
}

impl<'a> Default for PyEmbeddingDefault {
    fn default() -> Self {
        PyEmbeddingDefault::None
    }
}

impl<'a> FromPyObject<'a> for PyEmbeddingDefault {
    fn extract(ob: &'a PyAny) -> Result<Self, PyErr> {
        if ob.is_none() {
            return Ok(PyEmbeddingDefault::None);
        }
        if let Ok(emb) = ob
            .extract()
            .map(|e: &PyArray1<f32>| PyEmbeddingDefault::Embedding(e.to_owned()))
        {
            return Ok(emb);
        }

        if let Ok(constant) = ob.extract().map(PyEmbeddingDefault::Constant) {
            return Ok(constant);
        }
        if let Ok(embed) = ob
            .iter()
            .and_then(|iter| collect_array_from_py_iter(iter, ob.len().ok()))
            .map(PyEmbeddingDefault::Embedding)
        {
            return Ok(embed);
        }

        Err(exceptions::TypeError::py_err(
            "failed to construct default value.",
        ))
    }
}

fn collect_array_from_py_iter(iter: PyIterator, len: Option<usize>) -> PyResult<Py<PyArray1<f32>>> {
    let mut embed_vec = len.map(Vec::with_capacity).unwrap_or_default();
    for item in iter {
        let item = item.and_then(|item| item.extract())?;
        embed_vec.push(item);
    }
    let gil = Python::acquire_gil();
    let embed = PyArray1::from_vec(gil.python(), embed_vec).to_owned();
    Ok(embed)
}

struct Skips<'a>(HashSet<&'a str>);

impl<'a> FromPyObject<'a> for Skips<'a> {
    fn extract(ob: &'a PyAny) -> Result<Self, PyErr> {
        let mut set = ob.len().map(HashSet::with_capacity).unwrap_or_default();
        if ob.is_none() {
            return Ok(Skips(set));
        }
        for el in ob
            .iter()
            .map_err(|_| exceptions::TypeError::py_err("Iterable expected"))?
        {
            let el = el?;
            set.insert(el.extract().map_err(|_| {
                exceptions::TypeError::py_err(format!("Expected String not: {}", el))
            })?);
        }
        Ok(Skips(set))
    }
}

struct PyEmbedding<'a>(&'a PyArray1<f32>);

impl<'a> FromPyObject<'a> for PyEmbedding<'a> {
    fn extract(ob: &'a PyAny) -> Result<Self, PyErr> {
        let embedding = ob
            .downcast_ref::<PyArray1<f32>>()
            .map_err(|_| exceptions::TypeError::py_err("Expected array with dtype Float32"))?;
        if embedding.data_type() != NpyDataType::Float32 {
            return Err(exceptions::TypeError::py_err(format!(
                "Expected dtype Float32, got {:?}",
                embedding.data_type()
            )));
        };
        Ok(PyEmbedding(embedding))
    }
}
