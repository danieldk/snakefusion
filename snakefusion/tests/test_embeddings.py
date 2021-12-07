import numpy as np
import pytest

from snakefusion import Embeddings

TEST_NORMS = [
    6.557438373565674,
    8.83176040649414,
    6.164413928985596,
    9.165151596069336,
    7.4833149909973145,
    7.211102485656738,
    7.4833149909973145,
]


def test_embeddings(embeddings_fifu, embeddings_text, embeddings_text_dims):
    # Check that we cover all words from all embedding below.
    assert len(embeddings_fifu.vocab) == 7
    assert len(embeddings_text.vocab) == 7
    assert len(embeddings_text_dims.vocab) == 7
    fifu_storage = embeddings_fifu.storage
    # Check that the finalfusion embeddings have the correct dimensionality
    # The correct dimensionality of the other embedding types is asserted
    # in the pairwise comparisons below.
    assert fifu_storage.shape == (7, 10)

    for embedding, storage_row in zip(embeddings_fifu, fifu_storage):
        assert np.allclose(
            embedding.embedding, embeddings_text[embedding.word]
        ), "FiFu and text embedding mismatch"
        assert np.allclose(
            embedding.embedding, embeddings_text_dims[embedding.word]
        ), "FiFu and textdims embedding mismatch"
        assert np.allclose(
            embedding.embedding, storage_row
        ), "FiFu and storage row  mismatch"


def test_embedding_batch(embeddings_fifu):
    words = ["two", "one", "seven", "one", "three", "twenty", "two"]
    zeros = np.zeros((embeddings_fifu.storage.shape[1],))
    check_embeds = np.stack(
        [embeddings_fifu.embedding(word, default=zeros) for word in words]
    )

    # Check returning a fresh embedding matrix.
    batch_embeds, _ = embeddings_fifu.embedding_batch(words)
    assert np.allclose(batch_embeds, check_embeds)

    # Check with an output matrix
    output_embeds = np.zeros(
        (len(words), embeddings_fifu.storage.shape[1]), dtype=np.float32
    )
    output_embeds_returned, _ = embeddings_fifu.embedding_batch(
        words, out=output_embeds
    )
    assert np.allclose(output_embeds, check_embeds)
    assert np.allclose(output_embeds_returned, check_embeds)


def test_to_bytes_from_bytes_roundtrip(embeddings_fifu):
    serialized = embeddings_fifu.to_bytes()
    deserialized = Embeddings.from_bytes(serialized)

    for embed in embeddings_fifu:
        np.allclose(deserialized[embed.word], embed.embedding)


def test_unknown_embeddings(embeddings_fifu):
    assert (
        embeddings_fifu.embedding("OOV") is None
    ), "Unknown lookup with no default failed"
    assert (
        embeddings_fifu.embedding("OOV", default=None) is None
    ), "Unknown lookup with 'None' default failed"
    assert np.allclose(
        embeddings_fifu.embedding("OOV", default=[10] * 10), np.array([10.0] * 10)
    ), "Unknown lookup with 'list' default failed"
    assert np.allclose(
        embeddings_fifu.embedding("OOV", default=np.array([10.0] * 10)),
        np.array([10.0] * 10),
    ), "Unknown lookup with array default failed"
    assert np.allclose(
        embeddings_fifu.embedding("OOV", default=10), np.array([10.0] * 10)
    ), "Unknown lookup with 'int' scalar default failed"
    assert np.allclose(
        embeddings_fifu.embedding("OOV", default=10.0), np.array([10.0] * 10)
    ), "Unknown lookup with 'float' scalar default failed"
    with pytest.raises(TypeError):
        embeddings_fifu.embedding(
            "OOV", default="not working"
        ), "Unknown lookup with 'str' default succeeded"
    with pytest.raises(ValueError):
        embeddings_fifu.embedding(
            "OOV", default=[10.0] * 5
        ), "Unknown lookup with incorrectly shaped 'list' default succeeded"
    with pytest.raises(ValueError):
        embeddings_fifu.embedding(
            "OOV", default=np.array([10.0] * 5)
        ), "Unknown lookup with incorrectly shaped array default succeeded"
    with pytest.raises(ValueError):
        embeddings_fifu.embedding(
            "OOV", default=range(7)
        ), "Unknown lookup with iterable default with incorrect number succeeded"


def test_embeddings_pq(similarity_fifu, similarity_pq):
    for embedding in similarity_fifu:
        embedding_pq = similarity_pq.embedding("Berlin")
        assert np.allclose(
            embedding.embedding, embedding_pq, atol=0.3
        ), "Embedding and quantized embedding mismatch"


def test_embeddings_pq_mmap(similarity_fifu, similarity_pq_mmap):
    for embedding in similarity_fifu:
        embedding_pq = similarity_pq_mmap.embedding("Berlin")
        assert np.allclose(
            embedding.embedding, embedding_pq, atol=0.3
        ), "Embedding and quantized embedding mismatch"


def test_can_read_floret(embeddings_floret_check, embeddings_floret_text):
    for embed in embeddings_floret_check:
        assert np.allclose(
            embeddings_floret_text[embed.word], embed.embedding, atol=1e-4
        ), "Floret and floret check embeddings mismatch"


def test_embeddings_with_norms_oov(embeddings_fifu):
    assert embeddings_fifu.embedding_with_norm("Something out of vocabulary") is None


def test_indexing(embeddings_fifu):
    assert embeddings_fifu["one"] is not None
    with pytest.raises(KeyError):
        embeddings_fifu["Something out of vocabulary"]


def test_embeddings_oov(embeddings_fifu):
    assert embeddings_fifu.embedding("Something out of vocabulary") is None


def test_norms(embeddings_fifu):
    for embedding, norm in zip(embeddings_fifu, TEST_NORMS):
        assert pytest.approx(embedding.norm) == norm, "Norm fails to match!"
