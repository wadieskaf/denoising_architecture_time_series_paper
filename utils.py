import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import List, Tuple, Any


def plot_all_files(folder, file_type):
    """
    Plot all files in a folder.
    :param folder: path to folder
    :param file_type: file type to plot
    :return: figure
    """
    glob_string = os.path.join(folder, '*.' + file_type)
    csv_files = glob.glob(glob_string)
    num_files = len(csv_files)
    num_rows = np.ceil(num_files / 2).astype(int)
    fig, axs = plt.subplots(num_rows, 2, figsize=(20, num_rows * 6))
    for csv_file, ax in zip(csv_files, axs.ravel()):
        df = pd.read_csv(csv_file, index_col=0)
        title = os.path.basename(csv_file)
        df.plot(y='value', title=f'{title} Plot', use_index=True, ax=ax)

    return fig


def flatten_list(list_of_lists: List[List[Any]]) -> List[Any]:
    """
    Flatten a list of lists.
    :param list_of_lists: list of lists
    :return: flattened list
    """
    if list_of_lists == []:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten_list(list_of_lists[0]) + flatten_list(list_of_lists[1:])
    return list_of_lists[:1] + flatten_list(list_of_lists[1:])


def flatten_list_of_np_arrays(list_of_np_arrays: List[np.ndarray]) -> np.ndarray:
    """
    Flatten a list of numpy arrays.
    :param list_of_np_arrays: list of numpy arrays
    :return: flattened np array
    """
    return np.concatenate(list_of_np_arrays).ravel()


def get_data_stats_single(data: np.array) -> namedtuple:
    """
    Get statistics for data.
    :param data: data to get statistics for
    :return: namedtuple with statistics
    """
    data_stats = namedtuple('data_stats', ['median', 'mean', 'std', 'min', 'max'])
    return data_stats(np.median(data), np.mean(data), np.std(data), np.min(data), np.max(data))


def get_data_stats_list(data: List[np.array]) -> namedtuple:
    """
    Get statistics for data.
    :param data: data to get statistics for
    :return: namedtuple with statistics
    """
    f_data = flatten_list_of_np_arrays(data)
    return get_data_stats_single(f_data)


def normalize_data_single(data: np.ndarray, normalize_range: Tuple[int, int] = (-1, 1)) -> np.ndarray:
    """
    Normalize data.
    :param normalize_range: range to normalize to
    :param data: data to normalize
    :return: normalized data
    """
    data_stats = get_data_stats_single(data)
    a = normalize_range[0]
    b = normalize_range[1]
    min_data = data_stats.min
    max_data = data_stats.max
    normalized_data = (data - min_data) / (max_data - min_data) * (b - a) + a
    return normalized_data


def normalize_data_all(data: List[np.array], normalize_range: Tuple[int, int] = (-1, 1)) -> List[np.ndarray]:
    data_stats = get_data_stats_list(data)
    a = normalize_range[0]
    b = normalize_range[1]
    min_data = data_stats.min
    max_data = data_stats.max
    normalized_data = [(x - min_data) / (max_data - min_data) * (b - a) + a for x in data]
    return normalized_data


def normalize_data_list_one_per_time(data: List[np.ndarray], normalize_range: Tuple[int, int] = (-1, 1)) -> List[
    np.ndarray]:
    """
    Normalize data.
    :param normalize_range: range to normalize to
    :param data: data to normalize
    :return: normalized data
    """
    result = list()
    for d in data:
        d_stats = get_data_stats_single(d)
        a = normalize_range[0]
        b = normalize_range[1]
        result.append((d - d_stats.min) / (d_stats.max - d_stats.min) * (b - a) + a)
    return result


def load_json(json_file: str) -> dict:
    """
    Load json file.
    :param json_file: json file to load
    :return: json file
    """
    with open(json_file, 'r') as f:
        return json.load(f)


def list_to_string(list_of_strings: List[int]) -> str:
    """
    Convert list of strings to string.
    :param list_of_strings: list of integers
    :return: string
    """
    return ', '.join(map(str, list_of_strings)).strip()


def fix_csv_file(csv_file: str, target_column: str) -> None:
    """
    Fix csv file.
    :param csv_file: csv file to fix
    :param target_column: target column
    :return: fixed csv file
    """
    df = pd.read_csv(csv_file)
    temp_string = list_to_string(df[target_column].values)
    df = df.head(1)
    df[target_column] = temp_string
    df.to_csv(csv_file, index=False)


def str_to_list_of_int(string: str) -> List[int]:
    """
    Convert string to list of integers.
    :param string: string to convert
    :return: list of integers
    """
    return list(map(int, string.split(',')))

