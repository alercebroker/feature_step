import logging
import warnings
import numpy as np
import pandas as pd

from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from lc_classifier.features.preprocess import ElasticcPreprocessor
from lc_classifier.features import ElasticcFeatureExtractor

warnings.filterwarnings("ignore")
logging.getLogger("GP").setLevel(logging.WARNING)


class FeaturesComputer(GenericStep):
    """FeaturesComputer Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(
        self,
        consumer=None,
        config=None,
        producer=None,
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(consumer, config=config, level=level)
        self.preprocessor = ElasticcPreprocessor()
        self.features_computer = ElasticcFeatureExtractor()

        prod_config = self.config.get("PRODUCER_CONFIG", None)
        if prod_config:
            self.producer = producer or KafkaProducer(prod_config)
        else:
            self.producer = None

        self._rename_cols = {
            "mag": "FLUXCAL",
            "e_mag": "FLUXCALERR",
            "fid": "BAND",
            "mjd": "MJD"
        }

        self._fid_mapper = {
            0: "u",
            1: "g",
            2: "r",
            3: "i",
            4: "z",
            5: "Y",
        }

    def compute_features(self, detections: pd.DataFrame, metadata: pd.DataFrame):
        """Compute Hierarchical-Features in detections and non detections to `dict`.

        **Example:**

        Parameters
        ----------
        detections : pandas.DataFrame
            Detections of an object
        metadata: pd.DataFrame
            Metadata from detections
        """
        clean_lightcurves = self.preprocessor.preprocess(detections)
        features = self.features_computer.compute_features(clean_lightcurves, metadata=metadata)
        return features

    def produce(self, output_messages):
        for message in output_messages:
            aid = message["aid"]
            self.producer.produce(message, key=aid)

    def map_detections(self, light_curves: pd.DataFrame) -> pd.DataFrame:
        light_curves.drop(columns=["meanra", "meandec", "ndet", "non_detections", "metadata"], inplace=True)
        exploded = light_curves.explode("detections")
        detections = pd.DataFrame.from_records(exploded["detections"].values, index=exploded.index)
        detections = detections[self._rename_cols.keys()]
        detections = detections.rename(columns=self._rename_cols)
        detections["BAND"] = detections["BAND"].map(lambda x: self._fid_mapper[x])
        return detections

    def get_metadata(self, light_curves: pd.DataFrame):
        def _get_metadata(detections: pd.DataFrame):
            metadata = {
                "MWEBV": np.nan,
                "REDSHIFT_HELIO": np.nan
            }
            for det in detections:
                extra_fields = det["extra_fields"]
                if "diaObject" in extra_fields.keys():
                    dia_object = extra_fields["diaObject"]
                    if dia_object:
                        metadata = {
                            "MWEBV": dia_object["mwebv"],
                            "REDSHIFT_HELIO": dia_object["z_final"]
                        }
                        break
            return pd.Series(metadata)

        metadata_df = light_curves["detections"].apply(_get_metadata)
        return metadata_df

    def prepare_message(self, features, light_curves):
        features.replace({np.nan: None}, inplace=True)
        features_pack = pd.DataFrame({"features": features.to_dict("records")}, index=features.index)
        output_df = light_curves.join(features_pack)
        output_df.reset_index(inplace=True, drop=False)
        output_df["candid"] = output_df["candid"].astype(int)
        output_df.replace({np.nan: None}, inplace=True)
        output_messages = output_df.to_dict("records")
        return output_messages

    def execute(self, messages):
        light_curves_dataframe = pd.DataFrame(messages)
        light_curves_dataframe.drop_duplicates(subset="aid", inplace=True, keep="last")
        light_curves_dataframe.set_index("aid", inplace=True)
        self.logger.info(f"Processing {len(messages)} light curves.")
        detections = self.map_detections(light_curves_dataframe)
        self.logger.info(f"A total of {len(detections)} detections in {len(light_curves_dataframe)} light curves")
        metadata = self.get_metadata(light_curves_dataframe)
        features = self.compute_features(detections, metadata)
        self.logger.info(f"Features calculated: {len(features)}")
        output_messages = self.prepare_message(features, light_curves_dataframe)
        if self.producer:
            self.produce(output_messages)
