FEATURES = {
 "type": "map",
 "values": ["int", "null", "float", "boolean", "double"],
}

LIGHTCURVE = {
    "type": "array",
    "items": {
        "name": "detections_record",
        "type": "record",
        "fields": [
            {"name": "mjd", "type": "double"},
            {"name": "fid", "type": "int"},
            {"name": "mag", "type": "float"},
            {"name": "e_mag", "type": "float"},
        ],
    },
}


SCHEMA = {
    "doc": "Light curve with features",
    "name": "lcs_features",
    "type": "record",
    "fields": [
        {"name": "aid", "type": "string"},
        {"name": "alertId", "type": "string"},
        {"name": "candid", "type": "long"},
        {"name": "features", "type": [FEATURES, "null"]},
        {"name": "detections", "type": LIGHTCURVE},
        {"name": "elasticcPublishTimestamp", "type": "long"},
        {"name": "brokerIngestTimestamp", "type": "long"}
    ],
}
