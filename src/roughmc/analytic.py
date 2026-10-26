"""Closed-form prices used as benchmarks for the Monte Carlo estimators."""

import numpy as np
from scipy.stats import norm


def black_scholes_call(
    s0: float, strike: float, maturity: float, rate: float, sigma: float
) -> float:
    """Black-Scholes price of a European call option.

    Args:
        s0: Spot price of the underlying.
        strike: Strike price.
        maturity: Time to maturity, in years.
        rate: Continuously compounded risk-free rate.
        sigma: Volatility of the underlying.

    Returns:
        The price of the call.
    """
    d1 = (np.log(s0 / strike) + (rate + sigma**2 / 2) * maturity) / (
        sigma * np.sqrt(maturity)
    )
    d2 = d1 - sigma * np.sqrt(maturity)

    return s0 * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)
