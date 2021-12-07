use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor};
use std::sync::{Arc, RwLock};

use finalfusion::compat::floret::ReadFloretText;
use finalfusion::compat::text::{ReadText, ReadTextDims};
use finalfusion::compat::word2vec::ReadWord2Vec;
use finalfusion::metadata::Metadata;
use finalfusion::prelude::*;
use finalfusion::similarity::*;
use finalfusion::storage::{NdArray, Storage};
use finalfusion::vocab::Vocab;
use itertools::Itertools;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::class::iter::PyIterProtocol;
use pyo3::exceptions;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyIterator, PyTuple};
use reductive::pq::Pq;
#[cfg(feature = "opq")]
use reductive::pq::{GaussianOpq, Opq};
use toml::{self, Value};

use crate::storage::PyStorage;
use crate::{EmbeddingsWrap, PyEmbeddingIterator, PyVocab, PyWordSimilarity};

/// finalfusion embeddings.
#[pyclass(name = "Embeddings")]
pub struct PyEmbeddings {
    embeddings: Arc<RwLock<EmbeddingsWrap>>,
}

#[pymethods]
impl PyEmbeddings {
    /// new(path, mmap=False)
    /// --
    ///
    /// Load embeddings from the given `path`.
    ///
    /// When the `mmap` argument is `True`, the embedding matrix is
    /// not loaded into memory, but memory mapped. This results in
    /// lower memory use and shorter load times, while sacrificing
    /// some query efficiency.
    #[new]
    #[args(mmap = false)]
    fn new(path: &str, mmap: bool) -> PyResult<PyEmbeddings> {
        // First try to load embeddings with viewable storage. If that
        // fails, attempt to load the embeddings as non-viewable
        // storage.
        let embeddings = match read_embeddings(path, mmap) {
            Ok(e) => EmbeddingsWrap::View(e),
            Err(_) => read_embeddings(path, mmap)
                .map(EmbeddingsWrap::NonView)
                .map_err(|err| exceptions::PyIOError::new_err(err.to_string()))?,
        };

        Ok(PyEmbeddings {
            embeddings: Arc::new(RwLock::new(embeddings)),
        })
    }

    /// read_fasttext(path, /, lossy=False)
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

    /// read_floret_text(path)
    /// --
    ///
    /// Read embeddings in the floret text format.
    #[staticmethod]
    fn read_floret_text(path: &str) -> PyResult<PyEmbeddings> {
        read_non_fifu_embeddings(path, |r| Embeddings::read_floret_text(r))
    }

    /// read_text(path, /, lossy=False)
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

    /// read_text_dims(path, /, lossy=False)
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

    /// read_word2vec(path, /, lossy=False)
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

    fn __getitem__(&self, py: Python, word: &str) -> PyResult<Py<PyArray1<f32>>> {
        let embeddings = self.embeddings.read().unwrap();

        match py.allow_threads(|| embeddings.embedding(word)) {
            Some(embedding) => Ok(embedding.into_owned().into_pyarray(py).to_owned()),
            None => Err(exceptions::PyKeyError::new_err("Unknown word and n-grams")),
        }
    }

    /// Get the model's vocabulary.
    #[getter]
    fn vocab(&self) -> PyResult<PyVocab> {
        Ok(PyVocab::new(self.embeddings.clone()))
    }

    /// Get the model's storage.
    #[getter]
    fn storage(&self) -> PyStorage {
        PyStorage::new(self.embeddings.clone())
    }

    /// analogy(self, word1, word2, word3, limit=10, mask=(True, True, True))
    /// --
    ///
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
        let embeddings = self.embeddings.read().unwrap();

        let embeddings = embeddings.view().ok_or_else(|| {
            exceptions::PyValueError::new_err(
                "Analogy queries are not supported for this type of embedding matrix",
            )
        })?;

        let results = py
            .allow_threads(|| {
                embeddings.analogy_masked([word1, word2, word3], [mask.0, mask.1, mask.2], limit)
            })
            .map_err(|lookup| {
                let failed = [word1, word2, word3]
                    .iter()
                    .zip(lookup.iter())
                    .filter(|(_, success)| !*success)
                    .map(|(word, _)| word)
                    .join(" ");
                exceptions::PyKeyError::new_err(format!("Unknown word or n-grams: {}", failed))
            })?;

        Self::similarity_results(py, results)
    }

    /// embedding(self, word, /, default=None)
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
        py: Python,
        word: &str,
        default: PyEmbeddingDefault,
    ) -> PyResult<Option<Py<PyArray1<f32>>>> {
        let embeddings = self.embeddings.read().unwrap();
        if let PyEmbeddingDefault::Embedding(array) = &default {
            if array.as_ref(py).shape()[0] != embeddings.storage().shape().1 {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid shape of default embedding: {}",
                    array.as_ref(py).shape()[0]
                )));
            }
        }

        if let Some(embedding) = py.allow_threads(|| embeddings.embedding(word)) {
            return Ok(Some(embedding.into_owned().into_pyarray(py).to_owned()));
        };
        match default {
            PyEmbeddingDefault::Constant(constant) => {
                let nd_arr = Array1::from_elem([embeddings.storage().shape().1], constant);
                Ok(Some(nd_arr.into_pyarray(py).to_owned()))
            }
            PyEmbeddingDefault::Embedding(array) => Ok(Some(array)),
            PyEmbeddingDefault::None => Ok(None),
        }
    }

    /// embedding_batch(self, words, /, out=None)
    /// --
    ///
    /// Get the embedding for a batch of words. The embeddings are returned
    /// along with an array that indicates for each word whether an embedding
    /// could be found.
    ///
    /// If a matrix is provided through the `out` argument, embeddings are
    /// written to that matrix. Rows corresponding to words for which no
    /// embedding could be found are not overwritten.
    fn embedding_batch<'py>(
        &self,
        py: Python<'py>,
        words: Vec<&str>,
        out: Option<&'py PyArray2<f32>>,
    ) -> (&'py PyArray2<f32>, Vec<bool>) {
        let embeddings = self.embeddings.read().unwrap();

        let out = match out {
            Some(out) => out,
            None => PyArray2::zeros(py, (words.len(), embeddings.storage().shape().1), false),
        };

        let present = embeddings.embedding_batch_into(&words, unsafe { out.as_array_mut() });

        (out, present)
    }

    /// embedding_with_norm(self, word)
    /// --
    ///
    /// Look up the embedding and norm of a word. The embedding and
    /// norm are returned as a tuple.
    fn embedding_with_norm(&self, py: Python, word: &str) -> Option<Py<PyTuple>> {
        let embeddings = self.embeddings.read().unwrap();

        use EmbeddingsWrap::*;
        let embedding_with_norm = py.allow_threads(|| match &*embeddings {
            View(e) => e.embedding_with_norm(word),
            NonView(e) => e.embedding_with_norm(word),
        });

        embedding_with_norm.map(|e| {
            let embedding = e.embedding.into_owned().into_pyarray(py);
            (embedding, e.norm).into_py(py)
        })
    }

    /// from_bytes(data)
    /// --
    ///
    /// Deserialize embeddings from `bytes`.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let embeddings = EmbeddingsWrap::read_embeddings(&mut Cursor::new(data))?;
        Ok(PyEmbeddings {
            embeddings: Arc::new(RwLock::new(embeddings)),
        })
    }

    /// Embeddings metadata.
    #[getter]
    fn metadata(&self) -> PyResult<Option<String>> {
        let embeddings = self.embeddings.read().unwrap();

        use EmbeddingsWrap::*;
        let metadata = match &*embeddings {
            View(e) => e.metadata(),
            NonView(e) => e.metadata(),
        };

        match metadata.map(|v| toml::ser::to_string_pretty(&**v)) {
            Some(Ok(toml)) => Ok(Some(toml)),
            Some(Err(err)) => Err(exceptions::PyIOError::new_err(format!(
                "Metadata is invalid TOML: {}",
                err
            ))),
            None => Ok(None),
        }
    }

    /// quantize(self, n_subquantizers, /, quantizer="pq", n_subquantizer_bits=8, n_iterations=100, n_attempts=1, normalize=True)
    /// --
    ///
    /// Quantize the embeddings with the given hyperparemeters:
    ///
    /// * The number of subquantizers
    /// * The quantizer (``pq``, ``opq``, or ``gausian_opq``).
    /// * The number of bits per subquantizer.
    /// * The number of optimization iterations.
    /// * The number of quantization attempts per iteration.
    /// * Whether embeddings should be l2-normalized before quantization.
    ///
    /// Returns the quantized embeddings.
    #[allow(clippy::too_many_arguments)]
    #[args(
        quantizer = "\"pq\"",
        n_subquantizer_bits = 8,
        n_iterations = 100,
        n_attempts = 1,
        normalize = true
    )]
    fn quantize(
        &self,
        py: Python,
        n_subquantizers: usize,
        quantizer: &str,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        normalize: bool,
    ) -> PyResult<PyEmbeddings> {
        self.quantize_(
            py,
            quantizer,
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            normalize,
        )
    }

    /// Set the metadata of the embeddings.
    ///
    /// Must be a valid TOML.
    #[setter]
    fn set_metadata(&mut self, metadata: &str) -> PyResult<()> {
        let value = match metadata.parse::<Value>() {
            Ok(value) => value,
            Err(err) => {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Metadata is invalid TOML: {}",
                    err
                )));
            }
        };

        let mut embeddings = self.embeddings.write().unwrap();

        use EmbeddingsWrap::*;
        match &mut *embeddings {
            View(e) => e.set_metadata(Some(Metadata::new(value))),
            NonView(e) => e.set_metadata(Some(Metadata::new(value))),
        };

        Ok(())
    }

    /// to_bytes(self)
    /// --
    ///
    /// Serialize the embeddings to `bytes`.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        let embeddings = self.embeddings.read().unwrap();
        PyBytes::new_with(
            py,
            embeddings.write_embeddings_len(0) as usize,
            |bytes: &mut [u8]| {
                let mut cursor = Cursor::new(bytes);
                embeddings.write_embeddings(&mut cursor)
            },
        )
    }

    /// word_similarity(self, word, /, limit=10)
    /// --
    ///
    /// Perform a similarity query.
    #[args(limit = 10)]
    fn word_similarity(&self, py: Python, word: &str, limit: usize) -> PyResult<Vec<PyObject>> {
        let embeddings = self.embeddings.read().unwrap();

        let embeddings = embeddings.view().ok_or_else(|| {
            exceptions::PyValueError::new_err(
                "Similarity queries are not supported for this type of embedding matrix",
            )
        })?;

        let results = py
            .allow_threads(|| embeddings.word_similarity(word, limit))
            .ok_or_else(|| exceptions::PyKeyError::new_err("Unknown word and n-grams"))?;

        Self::similarity_results(py, results)
    }

    /// embedding_similarity(self, embeddings, /, skip=None, limit=10)
    /// --
    ///
    /// Perform a similarity query based on a query embedding. ``skip``
    /// specifies the set of words that should never be returned.
    #[args(limit = 10, skip = "Skips(HashSet::new())")]
    fn embedding_similarity(
        &self,
        py: Python,
        embedding: PyEmbedding,
        skip: Skips,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let embeddings = self.embeddings.read().unwrap();

        let embeddings = embeddings.view().ok_or_else(|| {
            exceptions::PyValueError::new_err(
                "Similarity queries are not supported for this type of embedding matrix",
            )
        })?;

        let embedding = embedding.0.as_array();

        if embedding.shape()[0] != embeddings.storage().shape().1 {
            return Err(exceptions::PyValueError::new_err(format!(
                "Incompatible embedding shapes: embeddings: ({},), query: ({},)",
                embedding.shape()[0],
                embeddings.storage().shape().1
            )));
        }

        let results = py
            .allow_threads(|| embeddings.embedding_similarity_masked(embedding, limit, &skip.0))
            .ok_or_else(|| exceptions::PyKeyError::new_err("Unknown word and n-grams"))?;

        Self::similarity_results(py, results)
    }

    /// write(self, filename)
    /// --
    ///
    /// Write the embeddings to a finalfusion file.
    fn write(&self, filename: &str) -> PyResult<()> {
        let embeddings = self.embeddings.read().unwrap();
        let f = File::create(filename)?;
        let mut writer = BufWriter::new(f);
        embeddings.write_embeddings(&mut writer)
    }
}

#[cfg(feature = "opq")]
#[allow(clippy::too_many_arguments)]
impl PyEmbeddings {
    fn quantize_(
        &self,
        py: Python,
        quantizer: &str,
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        normalize: bool,
    ) -> PyResult<PyEmbeddings> {
        let embeddings = self.embeddings.read().unwrap();

        let quantized_embeddings = match quantizer {
            "pq" => embeddings.quantize::<Pq<f32>>(
                py,
                n_subquantizers,
                n_subquantizer_bits,
                n_iterations,
                n_attempts,
                normalize,
            ),
            "opq" => embeddings.quantize::<Opq>(
                py,
                n_subquantizers,
                n_subquantizer_bits,
                n_iterations,
                n_attempts,
                normalize,
            ),
            "gaussian_opq" => embeddings.quantize::<GaussianOpq>(
                py,
                n_subquantizers,
                n_subquantizer_bits,
                n_iterations,
                n_attempts,
                normalize,
            ),
            quantizer => Err(PyValueError::new_err(format!(
                "Unsupported quantizer: {}, must be one of: pq, opq, gaussian_opq",
                quantizer
            ))),
        }?;

        Ok(PyEmbeddings {
            embeddings: Arc::new(RwLock::new(EmbeddingsWrap::NonView(
                quantized_embeddings.into(),
            ))),
        })
    }
}

#[cfg(not(feature = "opq"))]
#[allow(clippy::too_many_arguments)]
impl PyEmbeddings {
    fn quantize_(
        &self,
        py: Python,
        quantizer: &str,
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        normalize: bool,
    ) -> PyResult<PyEmbeddings> {
        let embeddings = self.embeddings.read().unwrap();
        let quantized_embeddings = match quantizer {
            "pq" => embeddings.quantize::<Pq<f32>>(
                py,
                n_subquantizers,
                n_subquantizer_bits,
                n_iterations,
                n_attempts,
                normalize,
            ),
            quantizer => Err(PyValueError::new_err(format!(
                "Unsupported quantizer: {}, opq and guassian_opq quantizers require LAPACK",
                quantizer
            ))),
        }?;

        Ok(PyEmbeddings {
            embeddings: Arc::new(RwLock::new(EmbeddingsWrap::NonView(
                quantized_embeddings.into(),
            ))),
        })
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
                    PyWordSimilarity::new(ws.word().to_owned(), ws.cosine_similarity()),
                )?,
                py,
            ))
        }
        Ok(r)
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

fn read_embeddings<S>(
    path: &str,
    mmap: bool,
) -> finalfusion::error::Result<Embeddings<VocabWrap, S>>
where
    Embeddings<VocabWrap, S>: ReadEmbeddings + MmapEmbeddings,
{
    let f = File::open(path).map_err(|e| {
        finalfusion::error::Error::read_error("Cannot open embeddings file for reading", e)
    })?;
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
    R: FnOnce(&mut BufReader<File>) -> finalfusion::error::Result<Embeddings<V, NdArray>>,
    V: Vocab,
    Embeddings<VocabWrap, StorageViewWrap>: From<Embeddings<V, NdArray>>,
{
    let f = File::open(path).map_err(|err| {
        exceptions::PyIOError::new_err(format!(
            "Cannot read text embeddings from '{}': {}'",
            path, err
        ))
    })?;
    let mut reader = BufReader::new(f);

    let embeddings = read_embeddings(&mut reader).map_err(|err| {
        exceptions::PyIOError::new_err(format!(
            "Cannot read text embeddings from '{}': {}'",
            path, err
        ))
    })?;

    Ok(PyEmbeddings {
        embeddings: Arc::new(RwLock::new(EmbeddingsWrap::View(embeddings.into()))),
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

        Err(exceptions::PyTypeError::new_err(
            "failed to construct default value.",
        ))
    }
}

fn collect_array_from_py_iter(
    iter: &PyIterator,
    len: Option<usize>,
) -> PyResult<Py<PyArray1<f32>>> {
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
            .map_err(|_| exceptions::PyTypeError::new_err("Iterable expected"))?
        {
            let el = el?;
            set.insert(el.extract().map_err(|_| {
                exceptions::PyTypeError::new_err(format!("Expected String not: {}", el))
            })?);
        }
        Ok(Skips(set))
    }
}

struct PyEmbedding<'a>(PyReadonlyArray1<'a, f32>);

impl<'a> FromPyObject<'a> for PyEmbedding<'a> {
    fn extract(ob: &'a PyAny) -> Result<Self, PyErr> {
        let embedding = ob
            .extract()
            .map_err(|_| exceptions::PyTypeError::new_err("Expected array with dtype Float32"))?;
        Ok(PyEmbedding(embedding))
    }
}
