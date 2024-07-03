from platform import system
import sys

import pandas as pd
import numpy as np
from os import path

THRESHOLD = 0.00006


def load_files(csv):
    return pd.read_csv(csv)


def check_struct(csv1, csv2):
    return csv1.shape == csv2.shape


def check_features(csv1, csv2):
    return list(csv1) == list(csv2)


def set_threshold(t):
    global THRESHOLD
    THRESHOLD = t


def check_values(csv1, csv2):
    for i in range(0, csv1.shape[0]):
        for j in range(0, csv1.shape[1]):
            val1 = csv1.iloc[i].iloc[j]
            val2 = csv2.iloc[i].iloc[j]
            if val1 != val2:
                if isinstance(val1, str) or isinstance(val2, str):
                    return False
                if np.isnan(val1):
                    if np.isnan(val2):
                        continue
                    else:
                        return False
                if np.isnan(val2):
                    return False
                if isinstance(val1, np.float64) and isinstance(val2, np.float64):
                    diff = abs(abs(val1) - abs(val2))
                    max_val = max(abs(val1), abs(val2))
                    diff_ratio = diff / max_val
                    if diff_ratio >= THRESHOLD:
                        return False
                    else:
                        continue
                else:
                    return False
    return True


def compare_csv(csv1, csv2):
    csv1 = load_files(csv1)
    csv2 = load_files(csv2)
    if not check_struct(csv1, csv2):
        return False
    if not check_features(csv1, csv2):
        return False
    if not check_values(csv1, csv2):
        return False
    return True


def main():
    pass
    #return comapre_csv(path.join(old_path, file), path.join(new_path, file))


if __name__ == "__main__":
    main()
