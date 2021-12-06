Quickstart
==========

Loading embeddings
------------------

Embeddings are loaded as follows:

.. code-block:: python

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


Queries
-------

With a set of embeddings loaded, you can look up an embedding or
perform similarity/analogy queries:

.. code-block:: python

    # Look up the embedding for 'Tübingen'
    embed = embeds.embedding("Tübingen")

    # Similarity query for "Tübingen"
    embeds.word_similarity("Tübingen")

    # Similarity query based on a vector, returning the closest embedding to
    # the input vector, skipping "Tübingen".
    embeds.embedding_similarity(embed, skip={"Tübingen"})

    # Default analogy query (Berlin is to Germany as Amsterdam is to ...)
    embeds.analogy("Berlin", "Deutschland", "Amsterdam")

    # Analogy query allowing "Deutschland" as answer.
    embeds.analogy("Berlin", "Deutschland", "Amsterdam", mask=(True,False,True))

Low-level data structures
-------------------------

If you want to operate directly on the full embedding matrix, you can get a copy
of this matrix through:

.. code-block:: python

    # get copy of embedding matrix, changes to this won't touch the original matrix
    embeds.storage.matrix_copy()

You can also use the vocabulary directly:

.. code-block:: python

    vocab = embeds.vocab

    # get a list of indices associated with "Tübingen"
    vocab.["Tübingen"]

    # get a list of `(ngram, index)` tuples for "Tübingen"
    vocab.ngram_indices("Tübingen")

    # get a list of subword indices for "Tübingen"
    v.subword_indices("Tübingen")

More usage examples can be found in the
[examples](https://github.com/finalfusion/finalfusion-python/tree/master/examples)
directory.
