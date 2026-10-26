import pytest

from roughmc.analytic import black_scholes_call


def test_black_scholes_reference_value() -> None:
    price = black_scholes_call(100, 100, 1, 0.05, 0.2)
    assert price == pytest.approx(10.450583572185565, rel=1e-12)
