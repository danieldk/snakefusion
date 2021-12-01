import os

import snakefusion
import pytest


@pytest.fixture
def analogy_fifu(tests_root):
    yield snakefusion.Embeddings(os.path.join(tests_root, "analogy.fifu"))


@pytest.fixture
def embeddings_fifu(tests_root):
    yield snakefusion.Embeddings(os.path.join(tests_root, "embeddings.fifu"))


@pytest.fixture
def embeddings_text(tests_root):
    yield snakefusion.Embeddings.read_text(os.path.join(tests_root, "embeddings.txt"))


@pytest.fixture
def embeddings_floret_check(tests_root):
    yield snakefusion.Embeddings.read_text_dims(
        os.path.join(tests_root, "floret-check.txt")
    )


@pytest.fixture
def embeddings_floret_text(tests_root):
    yield snakefusion.Embeddings.read_floret_text(
        os.path.join(tests_root, "embeddings.floret")
    )


@pytest.fixture
def similarity_fifu(tests_root):
    yield snakefusion.Embeddings(os.path.join(tests_root, "similarity.fifu"))


@pytest.fixture
def similarity_pq(tests_root):
    yield snakefusion.Embeddings(os.path.join(tests_root, "similarity-pq.fifu"))


@pytest.fixture
def similarity_pq_mmap(tests_root):
    yield snakefusion.Embeddings(
        os.path.join(tests_root, "similarity-pq.fifu"), mmap=True
    )


@pytest.fixture
def subword_fifu(tests_root):
    yield snakefusion.Embeddings(os.path.join(tests_root, "subword.fifu"))


@pytest.fixture
def embeddings_text_dims(tests_root):
    yield snakefusion.Embeddings.read_text_dims(
        os.path.join(tests_root, "embeddings.dims.txt")
    )


@pytest.fixture
def tests_root():
    yield os.path.dirname(__file__)
