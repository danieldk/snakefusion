import pytest

ANALOGY_ORDER = [
    "Deutschland",
    "Westdeutschland",
    "Sachsen",
    "Mitteldeutschland",
    "Brandenburg",
    "Polen",
    "Norddeutschland",
    "Dänemark",
    "Schleswig-Holstein",
    "Österreich",
    "Bayern",
    "Thüringen",
    "Bundesrepublik",
    "Ostdeutschland",
    "Preußen",
    "Deutschen",
    "Hessen",
    "Potsdam",
    "Mecklenburg",
    "Niedersachsen",
    "Hamburg",
    "Süddeutschland",
    "Bremen",
    "Russland",
    "Deutschlands",
    "BRD",
    "Litauen",
    "Mecklenburg-Vorpommern",
    "DDR",
    "West-Berlin",
    "Saarland",
    "Lettland",
    "Hannover",
    "Rostock",
    "Sachsen-Anhalt",
    "Pommern",
    "Schweden",
    "Deutsche",
    "deutschen",
    "Westfalen",
]


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6, 16, 32, None])
def test_analogies(analogy_fifu, batch_size):
    for idx, analogy in enumerate(
        analogy_fifu.analogy("Paris", "Frankreich", "Berlin", 40, batch_size=batch_size)
    ):
        assert ANALOGY_ORDER[idx] == analogy.word

    assert (
        analogy_fifu.analogy(
            "Paris",
            "Frankreich",
            "Paris",
            1,
            (True, False, True),
            batch_size=batch_size,
        )[0].word
        == "Frankreich"
    )
    assert (
        analogy_fifu.analogy(
            "Paris", "Frankreich", "Paris", 1, (True, True, True), batch_size=batch_size
        )[0].word
        != "Frankreich"
    )
    assert (
        analogy_fifu.analogy(
            "Frankreich",
            "Frankreich",
            "Frankreich",
            1,
            (False, False, False),
            batch_size=batch_size,
        )[0].word
        == "Frankreich"
    )
    assert (
        analogy_fifu.analogy(
            "Frankreich",
            "Frankreich",
            "Frankreich",
            1,
            (False, False, True),
            batch_size=batch_size,
        )[0].word
        != "Frankreich"
    )

    with pytest.raises(ValueError):
        analogy_fifu.analogy(
            "Paris", "Frankreich", "Paris", 1, (True, True), batch_size=batch_size
        )
    with pytest.raises(ValueError):
        analogy_fifu.analogy(
            "Paris",
            "Frankreich",
            "Paris",
            1,
            (True, True, True, True),
            batch_size=batch_size,
        )
    with pytest.raises(KeyError):
        analogy_fifu.analogy(
            "Paris", "OOV", "Paris", 1, (True, True, True), batch_size=batch_size
        )
