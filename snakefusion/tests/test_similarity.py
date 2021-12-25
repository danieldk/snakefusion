import pytest
import numpy

SIMILARITY_ORDER_STUTTGART_10 = [
    "Karlsruhe",
    "Mannheim",
    "München",
    "Darmstadt",
    "Heidelberg",
    "Wiesbaden",
    "Kassel",
    "Düsseldorf",
    "Leipzig",
    "Berlin",
]


SIMILARITY_ORDER = [
    "Potsdam",
    "Hamburg",
    "Leipzig",
    "Dresden",
    "München",
    "Düsseldorf",
    "Bonn",
    "Stuttgart",
    "Weimar",
    "Berlin-Charlottenburg",
    "Rostock",
    "Karlsruhe",
    "Chemnitz",
    "Breslau",
    "Wiesbaden",
    "Hannover",
    "Mannheim",
    "Kassel",
    "Köln",
    "Danzig",
    "Erfurt",
    "Dessau",
    "Bremen",
    "Charlottenburg",
    "Magdeburg",
    "Neuruppin",
    "Darmstadt",
    "Jena",
    "Wien",
    "Heidelberg",
    "Dortmund",
    "Stettin",
    "Schwerin",
    "Neubrandenburg",
    "Greifswald",
    "Göttingen",
    "Braunschweig",
    "Berliner",
    "Warschau",
    "Berlin-Spandau",
]


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6, 16, 32, None])
def test_similarity_berlin_40(similarity_fifu, batch_size):
    for idx, sim in enumerate(
        similarity_fifu.word_similarity("Berlin", 40, batch_size=batch_size)
    ):
        assert SIMILARITY_ORDER[idx] == sim.word


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6, 16, 32, None])
def test_similarity_stuttgart_10(similarity_fifu, batch_size):
    for idx, sim in enumerate(
        similarity_fifu.word_similarity("Stuttgart", 10, batch_size=batch_size)
    ):
        assert SIMILARITY_ORDER_STUTTGART_10[idx] == sim.word


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6, 16, 32, None])
def test_embedding_similarity_stuttgart_10(similarity_fifu, batch_size):
    stuttgart = similarity_fifu.embedding("Stuttgart")
    sims = similarity_fifu.embedding_similarity(
        stuttgart, limit=10, batch_size=batch_size
    )
    assert sims[0].word == "Stuttgart"

    for idx, sim in enumerate(sims[1:]):
        assert SIMILARITY_ORDER_STUTTGART_10[idx] == sim.word

    for idx, sim in enumerate(
        similarity_fifu.embedding_similarity(
            stuttgart, skip={"Stuttgart"}, limit=10, batch_size=batch_size
        )
    ):
        assert SIMILARITY_ORDER_STUTTGART_10[idx] == sim.word


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6, 16, 32, None])
def test_embedding_similarity_incompatible_shapes(similarity_fifu, batch_size):
    incompatible_embed = numpy.ones(1, dtype=numpy.float32)
    with pytest.raises(ValueError):
        similarity_fifu.embedding_similarity(incompatible_embed, batch_size=batch_size)
