import numpy as np
import pandas as pd
from scipy import optimize
from numba import njit


@njit()
def sn_model(times, ampl, t0, gamma, beta, t_rise, t_fall):
    sigmoid_factor = 1.0 / 3.0
    t1 = t0 + gamma

    sigmoid = 1.0 / (1.0 + np.exp(-sigmoid_factor * (times - t1)))
    den = 1 + np.exp(-(times - t0) / t_rise)
    flux = ((1 - beta) * np.exp(-(times - t1) / t_fall) * sigmoid + (1. - beta * (times - t0) / gamma) * (1 - sigmoid))
    return flux * ampl / den


def fit_sn_model(time: np.ndarray, flux: np.ndarray, error: np.ndarray) -> pd.Series:
    time = time - np.min(time)
    imax = np.argmax(flux)

    fmax = flux[imax]
    # order: amplitude, t0, gamma, beta, t_rise, t_fall (lower first, upper second)
    bounds = [[fmax / 3, -50, 1, 0, 1, 1], [fmax * 3, 50, 100, 1, 100, 100]]
    guess = np.clip([1.2 * fmax, -5, np.max(time), 0.5, time[imax] / 2, 40], *bounds)

    try:
        params, *_ = optimize.curve_fit(sn_model, time, flux, p0=guess, bounds=bounds, ftol=guess[0] / 20)
    except (ValueError, RuntimeError, optimize.OptimizeWarning):
        try:
            params, *_ = optimize.curve_fit(sn_model, time, flux, p0=guess, bounds=bounds, ftol=guess[0] / 3)
        except (ValueError, RuntimeError, optimize.OptimizeWarning):
            params = np.full_like(guess, np.nan)

    prediction = sn_model(time, *params)
    n_dof = prediction.size - params.size
    chi_dof = np.nan if n_dof < 1 else np.sum((prediction - flux) ** 2 / (error + 0.01) ** 2) / n_dof
    return pd.Series([*params, chi_dof], index=["A", "t0", "gamma", "beta", "tau_raise", "tau_fall", "chi"])


def fit_sn_model_v2(time: np.ndarray, flux: np.ndarray, error: np.ndarray) -> pd.Series:
    time = time - np.min(time)
    imax = np.argmax(flux)

    fmax = flux[imax]
    # order: amplitude, t0, gamma, beta, t_rise, t_fall (lower first, upper second)
    bounds = [[fmax / 3, -50, 1, 0, 1, 1], [fmax * 3, 70, 100, 1, 100, 100]]
    guess = np.clip([1.2 * fmax, time[imax] * 2 / 3, time[imax], 0.5, time[imax] / 2, 50], *bounds)

    try:
        params, *_ = optimize.curve_fit(sn_model, time, flux, p0=guess, bounds=bounds, sigma=5 + error, ftol=1e-8)
    except (ValueError, RuntimeError, optimize.OptimizeWarning):
        try:
            params, *_ = optimize.curve_fit(sn_model, time, flux, p0=guess, bounds=bounds, sigma=5 + error, ftol=0.1)
        except (ValueError, RuntimeError, optimize.OptimizeWarning):
            params = np.full_like(guess, np.nan)

    prediction = sn_model(time, *params)
    n_dof = prediction.size - params.size
    chi_dof = np.nan if n_dof < 1 else np.sum((prediction - flux) ** 2 / (error + 0.01) ** 2) / n_dof
    return pd.Series([*params, chi_dof], index=["A", "t0", "gamma", "beta", "tau_raise", "tau_fall", "chi"])
