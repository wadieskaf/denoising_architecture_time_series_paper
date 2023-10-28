import json

DATASETS_ATTRIBUTES = {
    'A1Benchmark': {
        'value_column': 'value',
        'label_column': 'is_anomaly',
        'file_name_pattern': r'real_\d+.csv',
    },
    'A2Benchmark': {
        'value_column': 'value',
        'label_column': 'is_anomaly',
        'file_name_pattern': r'synthetic_\d+.csv',
    },
    'A3Benchmark': {
        'value_column': 'value',
        'label_column': 'anomaly',
        'file_name_pattern': r'A3Benchmark-TS\d+.csv',
    },
    'A4Benchmark': {
        'value_column': 'value',
        'label_column': 'anomaly',
        'file_name_pattern': r'A4Benchmark-TS\d+.csv',
    }
}

json.dump(DATASETS_ATTRIBUTES, open('datasets_attributes.json', 'w'), indent=2)
