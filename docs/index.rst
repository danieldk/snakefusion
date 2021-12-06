.. snakefusion documentation master file, created by
   sphinx-quickstart on Mon Dec  6 09:16:25 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to snakefusion's documentation!
=======================================

``snakefusion`` is a Python package for reading, writing, and using finalfusion,
fastText, floret, GloVe, and word2vec embeddings.  This package is a thin
wrapper around the Rust finalfusion_ crate.

``snakefusion`` supports the same types of embeddings as ``finalfusion``:

* Vocabulary:

  - No subwords
  - Subwords
* Embedding matrix:

  - Array
  - Memory-mapped
  - Quantized
* Format:

  - fastText
  - finalfusion
  - floret
  - GloVe
  - word2vec

.. _finalfusion: https://docs.rs/finalfusion/

.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 2

   quickstart
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
