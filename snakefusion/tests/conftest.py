from pathlib import Path

import snakefusion
import pytest


@pytest.fixture
def analogy_fifu(tests_root):
    yield snakefusion.Embeddings(tests_root.joinpath("analogy.fifu"))


@pytest.fixture
def embeddings_fifu(tests_root):
    yield snakefusion.Embeddings(tests_root.joinpath("embeddings.fifu"))


@pytest.fixture
def embeddings_text(tests_root):
    yield snakefusion.Embeddings.read_text(tests_root.joinpath("embeddings.txt"))


@pytest.fixture
def embeddings_floret_check(tests_root):
    yield snakefusion.Embeddings.read_text_dims(tests_root.joinpath("floret-check.txt"))


@pytest.fixture
def embeddings_floret_text(tests_root):
    yield snakefusion.Embeddings.read_floret_text(
        tests_root.joinpath("embeddings.floret")
    )


@pytest.fixture
def similarity_fifu(tests_root):
    yield snakefusion.Embeddings(tests_root.joinpath("similarity.fifu"))


@pytest.fixture
def similarity_pq(tests_root):
    yield snakefusion.Embeddings(tests_root.joinpath("similarity-pq.fifu"))


@pytest.fixture
def similarity_pq_mmap(tests_root):
    yield snakefusion.Embeddings(tests_root.joinpath("similarity-pq.fifu"), mmap=True)


@pytest.fixture
def subword_fifu(tests_root):
    yield snakefusion.Embeddings(tests_root.joinpath("subword.fifu"))


@pytest.fixture
def embeddings_text_dims(tests_root):
    yield snakefusion.Embeddings.read_text_dims(
        tests_root.joinpath("embeddings.dims.txt")
    )


@pytest.fixture
def tests_root():
    yield Path(__file__).parents[0]
