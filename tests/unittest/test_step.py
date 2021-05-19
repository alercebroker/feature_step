import unittest
from unittest import mock
from features.step import (
    FeaturesComputer,
    ZTFFeatureExtractor,
    SQLConnection,
    KafkaProducer,
    Feature,
    FeatureVersion,
)
import numpy as np
import pandas as pd
import os

FILE_PATH = os.path.dirname(__file__)


class MockSession:
    def commit(self):
        pass

    def add(self, model):
        pass


class StepTestCase(unittest.TestCase):
    def setUp(self):
        self.step_config = {
            "DB_CONFIG": {"SQL": {}},
            "PRODUCER_CONFIG": {"fake": "fake"},
            "FEATURE_VERSION": "v1",
            "STEP_METADATA": {
                "STEP_VERSION": "feature",
                "STEP_ID": "feature",
                "STEP_NAME": "feature",
                "STEP_COMMENTS": "feature",
                "FEATURE_VERSION": "1.0-test",
            },
        }
        self.mock_database_connection = mock.create_autospec(SQLConnection)
        self.mock_database_connection.engine = mock.Mock()
        self.mock_database_connection.session = mock.create_autospec(MockSession)
        self.ztf_feature_extractor = ZTFFeatureExtractor(
            bands=(1, 2), stream=True)
        self.mock_producer = mock.create_autospec(KafkaProducer)
        self.step = FeaturesComputer(
            config=self.step_config,
            feature_extractor=self.ztf_feature_extractor,
            db_connection=self.mock_database_connection,
            producer=self.mock_producer,
            test_mode=True,
        )

        self.detections = pd.read_csv(
            os.path.join(
                FILE_PATH, '../examples/100_objects_detections_corr.csv'),
            index_col="objectId")
        self.detections['corrected'] = True
        self.detections.index.name = 'oid'

        self.non_detections = pd.read_csv(
            os.path.join(
                FILE_PATH, '../examples/100_objects_non_detections.csv'),
            index_col="objectId")
        self.non_detections["mjd"] = self.non_detections.jd - 2400000.5
        self.non_detections.index.name = 'oid'

        self.xmatches = self.mock_xmatches(self.detections)
        self.metadata = self.mock_metadata(self.detections)

    def mock_xmatches(self, detections):
        oids = detections.index.unique()
        xmatches = pd.DataFrame(
            data=oids.values.reshape([-1, 1]),
            columns=['oid']
        )
        xmatches['W1mag'] = np.random.randn(len(oids)) + 17.0
        xmatches['W2mag'] = np.random.randn(len(oids)) + 17.0
        xmatches['W3mag'] = np.random.randn(len(oids)) + 17.0
        return xmatches

    def mock_metadata(self, detections):
        oids = detections.index.values
        metadata = pd.DataFrame(
            data=oids.reshape([-1, 1]),
            columns=['oid']
        )
        candid = [oid + '_' + str(i) for oid, i in zip(oids, range(len(oids)))]
        metadata['candid'] = candid
        metadata['sgscore1'] = np.random.rand(len(oids))
        return metadata

    def test_insert_step_metadata(self):
        self.step.insert_step_metadata()
        self.mock_database_connection.query().get_or_create.assert_called_once()

    def test_compute_features(self):
        step_features = self.step.compute_features(
            self.detections,
            self.non_detections,
            self.metadata,
            self.xmatches
        )
        self.assertIsInstance(step_features, pd.DataFrame)
        self.assertEqual(97, len(step_features))

    @mock.patch.object(pd, "read_sql")
    def test_get_on_db(self, read_sql):
        features = pd.DataFrame()
        self.step.get_on_db(features)
        read_sql.assert_called_once()

    def test_insert_feature_version(self):
        self.mock_database_connection.query().get_or_create.return_value = (
            "instance",
            "created",
        )
        self.step.insert_feature_version("preprocess_id")
        self.assertEqual(self.step.feature_version, "instance")

    @mock.patch.object(FeaturesComputer, "get_fid")
    def test_update_db_empty(self, get_fid):
        to_update = pd.DataFrame()
        out_columns = ["oid", "name", "value"]
        apply_get_fid = lambda x: get_fid(x)
        self.step.update_db(to_update, out_columns, apply_get_fid)
        self.step.db.engine.execute.assert_not_called()

    def test_update_db_not_empty(self):
        to_update = pd.DataFrame(
            {
                "oid": ["ZTF1"],
                "feature_1": [123],
                "feature_2": [np.nan],
                "not_a_feature": [-1],
            }
        )
        to_update.set_index("oid", inplace=True)
        out_columns = ["oid", "name", "value"]
        apply_get_fid = lambda x: self.step.get_fid(x)
        feature_version = mock.Mock()
        feature_version.version = "test"
        self.step.feature_version = feature_version
        expected = [
            {
                "_oid": "ZTF1",
                "_name": "feature",
                "_fid": 1,
                "_value": 123,
                "_version": "test",
            },
            {
                "_oid": "ZTF1",
                "_name": "feature",
                "_fid": 2,
                "_value": None,
                "_version": "test",
            },
            {
                "_oid": "ZTF1",
                "_name": "not_a_feature",
                "_fid": -99,
                "_value": -1,
                "_version": "test",
            },
        ]
        updated = self.step.update_db(to_update, out_columns, apply_get_fid)
        self.step.db.engine.execute.assert_called()
        self.assertEqual(updated, expected)

    @mock.patch.object(FeaturesComputer, "get_fid")
    def test_insert_db_empty(self, get_fid):
        to_insert = pd.DataFrame()
        out_columns = ["oid", "name", "value"]
        apply_get_fid = lambda x: get_fid(x)
        feature_version = mock.Mock()
        feature_version.version = "test"
        self.step.feature_version = feature_version
        self.step.update_db(to_insert, out_columns, apply_get_fid)

    def test_insert_db_not_empty(self):
        to_insert = pd.DataFrame(
            {
                "oid": ["ZTF1"],
                "feature_1": [123],
                "feature_2": [456],
                "not_a_feature": [-1],
            }
        )
        to_insert.set_index("oid", inplace=True)
        out_columns = ["oid", "name", "value"]
        apply_get_fid = lambda x: self.step.get_fid(x)
        feature_version = mock.Mock()
        feature_version.version = "test"
        self.step.feature_version = feature_version
        self.step.insert_db(to_insert, out_columns, apply_get_fid)
        dict_to_insert = [
            {
                "oid": "ZTF1",
                "fid": 1,
                "name": "feature",
                "value": 123,
                "version": "test",
            },
            {
                "oid": "ZTF1",
                "fid": 2,
                "name": "feature",
                "value": 456,
                "version": "test",
            },
            {
                "oid": "ZTF1",
                "fid": -99,
                "name": "not_a_feature",
                "value": -1,
                "version": "test",
            },
        ]
        self.step.db.query().bulk_insert.assert_called_with(dict_to_insert, Feature)

    @mock.patch.object(FeaturesComputer, "get_on_db")
    @mock.patch.object(FeaturesComputer, "update_db")
    @mock.patch.object(FeaturesComputer, "insert_db")
    def test_add_to_db_update(self, insert_db, update_db, get_on_db):
        result = pd.DataFrame(
            {
                "oid": ["ZTF1"],
                "feature_1": [123],
                "feature_2": [456],
                "not_a_feature": [-1],
            }
        )
        result.set_index("oid", inplace=True)
        get_on_db.return_value = ["ZTF1"]
        out_columns = ["oid", "name", "value"]
        apply_get_fid = lambda x: self.step.get_fid(x)
        self.step.add_to_db(result)
        update_db.assert_called_once()
        insert_db.assert_not_called()

    @mock.patch.object(FeaturesComputer, "get_on_db")
    @mock.patch.object(FeaturesComputer, "update_db")
    @mock.patch.object(FeaturesComputer, "insert_db")
    def test_add_to_db_insert(self, insert_db, update_db, get_on_db):
        result = pd.DataFrame(
            {
                "oid": ["ZTF1"],
                "feature_1": [123],
                "feature_2": [456],
                "not_a_feature": [-1],
            }
        )
        result.set_index("oid", inplace=True)
        get_on_db.return_value = []
        out_columns = ["oid", "name", "value"]
        apply_get_fid = lambda x: self.step.get_fid(x)
        self.step.add_to_db(result)
        insert_db.assert_called_once()
        update_db.assert_not_called()

    @mock.patch.object(FeaturesComputer, "get_on_db")
    @mock.patch.object(FeaturesComputer, "update_db")
    @mock.patch.object(FeaturesComputer, "insert_db")
    def test_add_to_db_empty(self, insert_db, update_db, get_on_db):
        result = pd.DataFrame()
        get_on_db.return_value = []
        out_columns = ["oid", "name", "value"]
        apply_get_fid = lambda x: self.step.get_fid(x)
        self.step.add_to_db(result)
        insert_db.assert_not_called()
        update_db.assert_not_called()

    def test_get_fid(self):
        feature = "W1"
        fid = self.step.get_fid(feature)
        self.assertEqual(fid, 0)
        feature = "g-r_max"
        fid = self.step.get_fid(feature)
        self.assertEqual(fid, 12)
        feature = "somefeature_2"
        fid = self.step.get_fid(feature)
        self.assertEqual(fid, 2)
        feature = "something_else"
        fid = self.step.get_fid(feature)
        self.assertEqual(fid, -99)
        feature = 123
        fid = self.step.get_fid(feature)
        self.assertEqual(fid, -99)

    def test_check_feature_name(self):
        name = self.step.check_feature_name("feature_1")
        self.assertEqual(name, "feature")
        name = self.step.check_feature_name("feature_one_two")
        self.assertEqual(name, "feature_one_two")

    @mock.patch.object(FeaturesComputer, "compute_features")
    @mock.patch.object(FeaturesComputer, "produce")
    @mock.patch.object(FeaturesComputer, "add_to_db")
    @mock.patch.object(FeaturesComputer, "insert_feature_version")
    def test_execute_with_producer(
        self, insert_feature_version, add_to_db, produce, compute_features
    ):
        messages = [
            {
                "oid": "ZTF1",
                "candid": 123,
                "detections": [{"candid": 123, "oid": "ZTF1", "mjd": 456, "fid": 1}]
                * 10,
                "non_detections": [],
                "xmatches": {"allwise": {"W1mag": 1, "W2mag": 2, "W3mag": 3}},
                "metadata": {"ps1": {"sgscore1": {}}},
                "preprocess_step_id": "preprocess",
            }
        ]
        mock_feature_version = mock.create_autospec(FeatureVersion)
        mock_feature_version.version = self.step_config["FEATURE_VERSION"]
        self.step.execute(messages)
        produce.assert_called_once()

    @mock.patch.object(FeaturesComputer, "compute_features")
    @mock.patch.object(FeaturesComputer, "produce")
    @mock.patch.object(FeaturesComputer, "add_to_db")
    @mock.patch.object(FeaturesComputer, "insert_feature_version")
    def test_execute_duplicates(
        self, insert_feature_version, add_to_db, produce, compute_features
    ):
        message1 = {
            "oid": "ZTF1",
            "candid": 123,
            "detections": [{"candid": 123, "oid": "ZTF1", "mjd": 456, "fid": 1}] * 10,
            "non_detections": [{"oid": "ZTF1", "fid": 1, "mjd": 456}],
            "xmatches": {"allwise": {"W1mag": 1, "W2mag": 2, "W3mag": 3}},
            "metadata": {"ps1": {"sgscore1": {}}},
            "preprocess_step_id": "preprocess",
        }
        message2 = {
            "oid": "ZTF1",
            "candid": 123,
            "detections": [{"candid": 123, "oid": "ZTF1", "mjd": 456, "fid": 1}] * 10,
            "non_detections": [{"oid": "ZTF1", "fid": 1, "mjd": 456}],
            "xmatches": {"allwise": {"W1mag": 1, "W2mag": 2, "W3mag": 3}},
            "metadata": {"ps1": {"sgscore1": {}}},
            "preprocess_step_id": "preprocess",
        }
        messages = [message1, message2]
        mock_feature_version = mock.create_autospec(FeatureVersion)
        mock_feature_version.version = self.step_config["FEATURE_VERSION"]
        self.step.execute(messages)
        produce.assert_called_once()

    def test_produce(self):
        alert_data = pd.DataFrame({"oid": ["OID", "OID"], "candid": [123, 123]})
        features = pd.DataFrame({"oid": ["OID"], "feature1": 1, "feature2": 2})
        features.set_index("oid", inplace=True)
        self.step.produce(features, alert_data)
        expected_message = {
            "features": {"feature1": 1, "feature2": 2},
            "oid": "OID",
            "candid": 123,
        }
        self.step.producer.produce.assert_called_with(expected_message, key="OID")

    def test_produce_with_candid_series(self):
        alert_data = pd.read_csv(
            FILE_PATH + "/../examples/alert_data_with_duplicates.csv"
        )
        features = pd.read_csv(
            FILE_PATH + "/../examples/features_with_candid_problem.csv"
        )
        features.set_index("oid", inplace=True)
        self.step.produce(features, alert_data)
