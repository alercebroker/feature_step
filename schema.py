FEATURES = {
 "type": "map",
 "values": ["int", "null", "float", "boolean", "double"],
}

EXTRA_FIELDS = {
    "type": "map",
    "values": [
        "string",
        "int",
        "null",
        "float",
        "boolean",
        "double",
        "bytes",
        {
            "type": "map",
            "values": ["string", "float", "null", "int"],
        },
    ],
}

DETECTIONS = {
    "type": "array",
    "items": {
        "name": "detections_record",
        "type": "record",
        "fields": [
            {"name": "mjd", "type": "double"},
            {"name": "fid", "type": "int"},
            {"name": "mag", "type": "float"},
            {"name": "e_mag", "type": "float"},
            {"name": "extra_fields", "type": EXTRA_FIELDS},
        ],
    },
}

NON_DETECTIONS = {
    "type": "array",
    "items": {
        "name": "non_detections_record",
        "type": "record",
        "fields": [
            {"name": "tid", "type": "string"},
            {"name": "oid", "type": "string"},
            {"name": "mjd", "type": "double"},
            {"name": "diffmaglim", "type": "float"},
            {"name": "fid", "type": "int"},
            {"name": "extra_fields", "type": [EXTRA_FIELDS, "null"]},
        ],
    },
}


SCHEMA = {
    "doc": "Light curve with features",
    "name": "lcs_features",
    "type": "record",
    "fields": [
        {"name": "aid", "type": "string"},
        {"name": "alertId", "type": "long"},
        {"name": "candid", "type": "long"},
        {"name": "features", "type": [FEATURES, "null"]},
        {"name": "detections", "type": DETECTIONS},
        {"name": "non_detections", "type": NON_DETECTIONS},
        {"name": "elasticcPublishTimestamp", "type": "long"},
        {"name": "brokerIngestTimestamp", "type": "long"}
    ],
}
