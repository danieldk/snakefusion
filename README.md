# 🐍 snakefusion

[![Documentation Status](https://readthedocs.org/projects/snakefusion/badge/?version=latest)](https://snakefusion.readthedocs.io/en/latest/?badge=latest)
[![pypi Version](https://img.shields.io/pypi/v/snakefusion.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/snakefusion/)

## Introduction

`snakefusion` is a Python package for reading, writing, and using finalfusion,
fastText, floret, GloVe, and word2vec embeddings.  This package is a thin
wrapper around the Rust [finalfusion](https://docs.rs/finalfusion/) crate.

`snakefusion` supports the same types of embeddings as `finalfusion`:

* Vocabulary:
  * No subwords
  * Subwords
* Embedding matrix:
  * Array
  * Memory-mapped
  * Quantized
* Format:
  * fastText
  * finalfusion
  * floret
  * GloVe
  * word2vec

## Building from source

Building `snakefusion` from source requires a Rust toolchain that is installed
through [rustup](https://rustup.rs) and `setuptools-rust`:

~~~shell
$ pip install --upgrade setuptools-rust
~~~

You can then build and install `snakefusion` in your environment:

~~~shell
$ pip install .
~~~

## Documentation

Read the [snakefusion documentation](https://snakefusion.readthedocs.io/) for a
quickstart and API reference.

You use [finalfrontier](https://finalfusion.github.io/finalfrontier) to train
new embeddings, or download some [pretrained
embeddings](https://finalfusion.github.io/pretrained).
