import numpy
import pytest

import snakefusion


def test_quantization(analogy_fifu):
    quantized = analogy_fifu.quantize(50, n_subquantizer_bits=5)
    for embed in analogy_fifu:
        assert numpy.allclose(quantized[embed.word], embed.embedding, atol=0.2)


@pytest.mark.skipif(
    snakefusion.Config.OPQ, reason="snakefusion is compiled with OPQ support"
)
def test_quantization_opq_fails_without_opq(analogy_fifu):
    with pytest.raises(ValueError, match="Unsupported quantizer"):
        analogy_fifu.quantize(50, quantizer="opq", n_subquantizer_bits=5)

    with pytest.raises(ValueError, match="Unsupported quantizer"):
        analogy_fifu.quantize(50, quantizer="gaussian_opq", n_subquantizer_bits=5)


@pytest.mark.skipif(
    not snakefusion.Config.OPQ, reason="snakefusion is not compiled with OPQ support"
)
def test_opq_quantization(analogy_fifu):
    opq_quantized = analogy_fifu.quantize(50, quantizer="opq", n_subquantizer_bits=5)
    for embed in analogy_fifu:
        assert numpy.allclose(opq_quantized[embed.word], embed.embedding, atol=0.2)

    gaussian_opq_quantized = analogy_fifu.quantize(
        50, quantizer="gaussian_opq", n_subquantizer_bits=5
    )
    for embed in analogy_fifu:
        assert numpy.allclose(
            gaussian_opq_quantized[embed.word], embed.embedding, atol=0.2
        )
