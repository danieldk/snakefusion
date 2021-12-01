# 🐍 snakefusion

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

## Usage

Embeddings can be loaded as follows:

~~~python
import snakefusion

# Loading embeddings in finalfusion format
embeds = snakefusion.Embeddings("myembeddings.fifu")

# Or if you want to memory-map the embedding matrix:
embeds = snakefusion.Embeddings("myembeddings.fifu", mmap=True)

# fastText format
embeds = snakefusion.Embeddings.read_fasttext("myembeddings.bin")

# floret format
embeds = snakefusion.Embeddings.read_floret_text("myembeddings.floret")

# word2vec format
embeds = snakefusion.Embeddings.read_word2vec("myembeddings.w2v")
~~~

You can then compute an embedding, perform similarity queries, or analogy
queries:

~~~python
e = embeds.embedding("Tübingen")

# default similarity query for "Tübingen"
embeds.word_similarity("Tübingen")

# similarity query based on a vector, returning the closest embedding to
# the input vector, skipping "Tübingen"
embeds.embeddings_similarity(e, skip={"Tübingen"})

# default analogy query
embeds.analogy("Berlin", "Deutschland", "Amsterdam")

# analogy query allowing "Deutschland" as answer
embeds.analogy("Berlin", "Deutschland", "Amsterdam", mask=(True,False,True))
~~~

If you want to operate directly on the full embedding matrix, you can get a copy
of this matrix through:

~~~python
# get copy of embedding matrix, changes to this won't touch the original matrix
e.matrix_copy()
~~~

Finally access to the vocabulary is provided through:

~~~python
v = e.vocab()
# get a list of indices associated with "Tübingen"
v.item_to_indices("Tübingen")

# get a list of `(ngram, index)` tuples for "Tübingen"
v.ngram_indices("Tübingen")

# get a list of subword indices for "Tübingen"
v.subword_indices("Tübingen")
~~~

More usage examples can be found in the
[examples](https://github.com/finalfusion/finalfusion-python/tree/master/examples)
directory.

## Where to go from here

  * [finalfrontier](https://finalfusion.github.io/finalfrontier)
  * [finalfusion](https://finalfusion.github.io/)
  * [pretrained embeddings](https://finalfusion.github.io/pretrained)
