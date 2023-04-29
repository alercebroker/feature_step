import methodtools
import numpy as np
import pandas as pd

from ..utils import fill_index
from ._base import BaseHandler


class DetectionsHandler(BaseHandler):
    INDEX = "candid"
    UNIQUE = ["id", "fid", "mjd"]
    _COLUMNS = BaseHandler._COLUMNS + ["mag", "e_mag", "mag_ml", "e_mag_ml", "isdiffpos", "ra", "dec"]

    def _post_process_alerts(self, **kwargs):
        if kwargs.pop("legacy", False):
            self._alerts["mag"] = self._alerts["magpsf"]
            self._alerts["e_mag"] = self._alerts["sigmapsf"]
            self._alerts["mag_corr"] = self._alerts["magpsf_corr"]
            self._alerts["e_mag_corr"] = self._alerts["sigmapsf_corr"]
            self._alerts["e_mag_corr_ext"] = self._alerts["sigmapsf_corr_ext"]
        if kwargs.pop("corr", False):
            self._use_corrected_magnitudes(kwargs.pop("surveys"))
        else:
            self._alerts = self._alerts.assign(mag_ml=self._alerts["mag"], e_mag_ml=self._alerts["e_mag"])
        super()._post_process_alerts(**kwargs)

    def _use_corrected_magnitudes(self, surveys: tuple[str, ...]):
        idx = self._alerts[self._surveys_mask(surveys)].groupby("id")["mjd"].idxmin()
        corrected = self._alerts["corrected"][idx].set_axis(idx.index).reindex(self._alerts["id"])

        mag = np.where(corrected, self._alerts["mag_corr"], self._alerts["mag"])
        e_mag = np.where(corrected, self._alerts["e_mag_corr_ext"], self._alerts["e_mag"])

        self._alerts = self._alerts.assign(mag_ml=mag, e_mag_ml=e_mag)

    @methodtools.lru_cache()
    def get_colors(
        self, func: str, bands: tuple[str, str], *, surveys: str | tuple[str, ...] = (), ml: bool = True
    ) -> pd.Series:
        first, second = bands

        mags = self.get_aggregate(f"mag{'_ml' if ml else ''}", func, by_fid=True, surveys=surveys, bands=bands)
        mags = fill_index(mags, fid=bands)
        return mags.xs(first, level="fid") - mags.xs(second, level="fid")

    @methodtools.lru_cache()
    def get_count_by_sign(self, sign: int, *, by_fid: bool = False, bands: str | tuple[str, ...] = ()) -> pd.Series:
        counts = self.get_aggregate("isdiffpos", "value_count", by_fid=by_fid, bands=bands)
        if by_fid and bands:
            return fill_index(counts, fill_value=0, fid=bands, isdiffpos=(-1, 1)).xs(sign, level="isdiffpos")
        return fill_index(counts, fill_value=0, isdiffpos=(-1, 1)).xs(sign, level="isdiffpos")
