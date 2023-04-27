from typing import Any

import numpy as np
import pandas as pd


def fill_index(df: pd.DataFrame | pd.Series, *, fill_value: Any = np.nan, **kwargs):
    if not kwargs:
        raise ValueError("At least one additional index must be included to fill in")
    if "aid" in kwargs:
        raise ValueError("The index `aid` cannot be pre-specified")
    check = ["aid"] + list(kwargs.keys())
    if set(df.index.names) != set(check):
        raise ValueError(f"DataFrame has MultiIndex with {df.index.names}. Requested filling: {check}")
    aids = df.index.get_level_values("aid").unique()
    values = [aids] + [kwargs[k] for k in df.index.names if k != "aid"]
    return df.reindex(pd.MultiIndex.from_product(values, names=df.index.names), fill_value=fill_value)