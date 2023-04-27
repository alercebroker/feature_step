import functools
from typing import Any

import pandas as pd

from . import functions


def columns_per_fid(method):
    """Decorated method must produce a multi-indexed data frame with `fid` as a named level"""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        df = method(self, *args, **kwargs).unstack("fid")
        if self.BANDS_MAPPING:
            df.rename(columns=self.BANDS_MAPPING, level="fid", inplace=True)
        df.columns = df.columns.map(lambda lvls: f"{'_'.join(str(lvl) for lvl in lvls)}")
        return df

    return wrapper


def add_fid(fid: Any):
    """Adds a sub-column called `fid` with the given value to all columns"""

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            df = method(self, *args, **kwargs)
            return df.assign(fid=fid).set_index("fid", append=True).unstack("fid")

        return wrapper

    return decorator


def fill_in_every_fid(*, fill_value: Any = pd.NA):
    """Decorated method must produce a multi-indexed data frame with two levels, `aid` and `fid` (in that order)"""

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            df = method(self, *args, **kwargs)
            if self.BANDS:
                return functions.fill_index(df, fill_value=fill_value, fid=self.BANDS)
            return df

        return wrapper

    return decorator