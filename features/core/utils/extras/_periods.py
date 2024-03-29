import functools

import numpy as np
import pandas as pd
from P4J import MultiBandPeriodogram

from ._kim import apply_kim, empty_kim
from ._harmonics import apply_harmonics, empty_harmonics
from ._power_rate import apply_power_rates, empty_power_rates


@functools.lru_cache
def _get_multiband_periodogram() -> MultiBandPeriodogram:
    """Creates the multi-band periodogram for computations."""
    return MultiBandPeriodogram(method="MHAOV")


@functools.lru_cache()
def _indices(fids: tuple[str, ...]) -> pd.MultiIndex:
    """Creates indices for basic period pandas series."""
    n_fids = len(fids)
    level1 = ("".join(fids),) * 2 + fids + fids
    level0 = (
        ("Multiband_period", "PPE")
        + ("Period_band",) * n_fids
        + ("delta_period",) * n_fids
    )
    return pd.MultiIndex.from_arrays([level0, level1], names=(None, "fid"))


def periods(
    df: pd.DataFrame,
    fids: tuple[str, ...],
    kim: bool = False,
    n_harmonics: int = 0,
    factors: tuple[str, ...] = (),
) -> pd.Series:
    """Compute period features for an object.

    Args:
        df: Frame with magnitudes, errors and times. Expects a single object
        fids: Bands required. If not present, it will fill features with NaN
        kim: Whether to compute phase-folded features
        n_harmonics: Number of harmonic series features to compute. Set to 0 to skip
        factors: Power rate factors to compute (as strings). Leave empty to skip

    Returns:
        pd.Series: Period related features
    """
    periodogram = _get_multiband_periodogram()

    # Periods are only computed for bands with more than 5, but phase-folded and harmonics features use all, always
    filtered_df = df.groupby("fid").filter(lambda x: len(x) > 5).sort_values("mjd")

    fid = filtered_df["fid"].values
    mag, e_mag, mjd = filtered_df[["mag_ml", "e_mag_ml", "mjd"]].T.values
    periodogram.set_data(mjd, mag, e_mag, fid)
    try:
        periodogram.frequency_grid_evaluation(fmin=1e-3, fmax=20.0, fresolution=1e-3)
        periodogram.finetune_best_frequencies(n_local_optima=10, fresolution=1e-4)
    except TypeError:
        output = [pd.Series(np.nan, index=_indices(fids))]
        if kim:
            output.append(empty_kim(fids))
        if n_harmonics:
            output.append(empty_harmonics(n_harmonics, fids))
        if factors:
            output.append(empty_power_rates(factors, fids))
        return pd.concat(output)

    (frequency, *_), _ = periodogram.get_best_frequencies()
    period = 1 / frequency

    n_best = 100
    freq, per = periodogram.get_periodogram()
    tops = np.sort(per)[-n_best:] + 1e-2
    tops /= np.sum(tops)
    significance = 1 - np.sum(-tops * np.log(tops)) / np.log(n_best)

    per_band = np.full(2 * len(fids), np.nan)
    for i, band in enumerate(fids):
        try:
            period_band = 1 / periodogram.get_best_frequency(band)
        except KeyError:
            continue
        per_band[i] = period_band
        per_band[i + len(fids)] = abs(period - period_band)

    output = [pd.Series([period, significance, *per_band], index=_indices(fids))]
    if kim:
        output.append(apply_kim(df, period, fids))
    if n_harmonics:
        output.append(apply_harmonics(df, n_harmonics, period, fids))
    if factors:
        output.append(apply_power_rates(freq, per, factors, fids))
    return pd.concat(output)
