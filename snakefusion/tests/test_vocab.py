import pytest

TEST_NGRAM_INDICES = [
    ("tüb", [14]),
    ("en>", [69]),
    ("übinge", [74]),
    ("gen", [124]),
    ("ing", [168]),
    ("ngen", [181]),
    ("bing", [197]),
    ("inge", [246]),
    ("übin", [250]),
    ("tübi", [276]),
    ("bingen", [300]),
    ("<tübin", [308]),
    ("bin", [325]),
    ("übing", [416]),
    ("gen>", [549]),
    ("ngen>", [590]),
    ("ingen>", [648]),
    ("tübing", [651]),
    ("übi", [707]),
    ("ingen", [717]),
    ("binge", [761]),
    ("<tübi", [817]),
    ("<tü", [820]),
    ("<tüb", [857]),
    ("nge", [860]),
    ("tübin", [1007]),
]


def test_get(embeddings_text_dims):
    vocab = embeddings_text_dims.vocab
    assert vocab.get("one") is 0


def test_get_oov(embeddings_fifu):
    vocab = embeddings_fifu.vocab
    assert vocab.get("Something out of vocabulary") is None


def test_get_oov_with_default(embeddings_fifu):
    vocab = embeddings_fifu.vocab
    assert vocab.get("Something out of vocabulary", default=-1) == -1


def test_ngram_indices(subword_fifu):
    vocab = subword_fifu.vocab
    ngram_indices = sorted(vocab.ngram_indices("tübingen"), key=lambda tup: tup[1])
    for ngram_index, test_ngram_index in zip(ngram_indices, TEST_NGRAM_INDICES):
        assert ngram_index == test_ngram_index


def test_subword_indices(subword_fifu):
    vocab = subword_fifu.vocab
    subword_indices = sorted(vocab.subword_indices("tübingen"))
    test_indices = sorted(
        [index for ngram_index in TEST_NGRAM_INDICES for index in ngram_index[1]]
    )
    assert subword_indices == test_indices


def test_int_idx(embeddings_text_dims):
    vocab = embeddings_text_dims.vocab
    assert vocab[0] == "one"


def test_int_idx_out_of_range(embeddings_text_dims):
    vocab = embeddings_text_dims.vocab
    with pytest.raises(IndexError):
        _ = vocab[42]


def test_negative_int_idx(embeddings_text_dims):
    vocab = embeddings_text_dims.vocab
    assert vocab[-1] == "seven"


def test_negative_int_idx_out_of_range(embeddings_text_dims):
    vocab = embeddings_text_dims.vocab
    with pytest.raises(IndexError):
        _ = vocab[-42]


def test_string_idx(embeddings_text_dims):
    vocab = embeddings_text_dims.vocab
    assert vocab["one"] == 0


def test_string_oov(embeddings_text_dims):
    vocab = embeddings_text_dims.vocab
    with pytest.raises(KeyError):
        vocab["definitely in vocab"]


def test_string_oov_subwords(subword_fifu):
    vocab = subword_fifu.vocab
    assert sorted(vocab["tübingen"]) == [
        index for ngram_index in TEST_NGRAM_INDICES for index in ngram_index[1]
    ]
