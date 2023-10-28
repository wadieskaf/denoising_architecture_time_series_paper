from CompactJSONEncoder import CompactJSONEncoder


def build_key_from_list(list_of_values):
    """
    Builds a key from a list of values.
    :param list_of_values: list of values
    :return: key
    """
    key = ""
    for value in list_of_values:
        key += str(value) + "_"
    return key[:-1]


SEQ_LENS = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
ONLY_SMALLER_ARCHITECTURES = True

ARCHITECTURES = [[256], [128], [64], [32], [16], [8], [4], [2],
                 [265, 128], [128, 64], [64, 32], [32, 16], [16, 8], [8, 4], [4, 2],
                 [256, 128, 64], [128, 64, 32], [64, 32, 16], [32, 16, 8], [16, 8, 4], [8, 4, 2],
                 [256, 128, 64, 32], [128, 64, 32, 16], [64, 32, 16, 8], [32, 16, 8, 4], [16, 8, 4, 2],
                 [256, 128, 64, 32, 16], [128, 64, 32, 16, 8], [64, 32, 16, 8, 4], [32, 16, 8, 4, 2],
                 [256, 128, 64, 32, 16, 8], [128, 64, 32, 16, 8, 4], [64, 32, 16, 8, 4, 2],
                 [256, 128, 64, 32, 16, 8, 4], [128, 64, 32, 16, 8, 4, 2],
                 [256, 128, 64, 32, 16, 8, 4, 2]]

exps = dict()

for architecture in ARCHITECTURES:
    exp = {
        "architecture": architecture,
        "seq_lens": [seq_len for seq_len in SEQ_LENS if
                     seq_len >= max(architecture)] if ONLY_SMALLER_ARCHITECTURES else SEQ_LENS,
        "dropouts": DROPOUTS,
    }
    if exp['seq_lens']:
        exps.update({build_key_from_list(architecture): exp})

encoder = CompactJSONEncoder(indent=1)

with open("exps.json", "w") as f:
    f.write(encoder.encode(exps))
