import random
import numpy as np
import tensorflow as tf
from bisect import bisect
from itertools import accumulate
from typing import List, Callable, Tuple
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, data: List[np.ndarray], batch_size: int = 8, seq_len: int = 100, step: int = 1, normalize=False,
                 normalize_function: Callable[[List, Tuple], List] = None, normalize_range: Tuple = (-1, 1),
                 test: bool = False, shuffle_data: bool = False, random_seed: int = 42):
        if normalize:
            self.data = normalize_function(data, normalize_range)
        else:
            self.data = data
        self.num_files = len(data)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.step = step
        self.test = test
        self.shuffle_data = shuffle_data
        self.random_seed = random_seed
        self.current_file = 0
        self.n = 0
        self.num_samples = self._get_num_of_samples_per_file()
        self.num_batches = self._get_num_batches_per_file()
        self.accum_num_batches = list(accumulate(self.num_batches))
        # set random seeds
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    def _get_single_ts_num_of_samples(self, index):
        return len(self.data[index])

    def _get_num_of_samples_per_file(self):
        return [self._get_single_ts_num_of_samples(i) for i in range(len(self.data))]

    def _single_ts_len(self, i):
        possible_steps = list(range(self.num_samples[i] - self.seq_len, 0, -self.step))
        possible_ends = [x + self.seq_len for x in possible_steps]
        valid_ends = list(filter(lambda x: x < self.num_samples[i], possible_ends))
        num_valid_ends = len(valid_ends)
        return np.ceil(num_valid_ends / float(self.batch_size)).astype(int)

    def _get_num_batches_per_file(self):
        return [self._single_ts_len(i) for i in range(len(self.data))]

    def __len__(self):
        return sum(self.num_batches)

    def __get_sample(self):
        sample = self.data[self.current_file][self.n: self.n + self.seq_len]
        return sample

    def _go_to_idx(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        if idx in self.accum_num_batches:
            self.current_file = self.accum_num_batches.index(idx)
            self.n = 0
        else:
            file_num = bisect(self.accum_num_batches, idx)
            self.current_file = file_num
            self.n = idx - self.accum_num_batches[file_num - 1] if file_num != 0 else idx

    def __getitem__(self, idx):
        self._go_to_idx(idx)
        result = list()
        for i in range(self.batch_size):
            item = self.__get_sample()
            result.append(item.tolist())
            self.n += self.step
        output = np.array(result)
        output = np.expand_dims(output, axis=2)
        if not self.test:
            return output
        else:
            return output, output

    def on_epoch_end(self):
        if self.shuffle_data:
            temp = list(zip(self.data, self.num_samples, self.num_batches))
            random.shuffle(temp)
            self.data, self.num_samples, self.num_batches = zip(*temp)
            self.accum_num_batches = list(accumulate(self.num_batches))


class DataGeneratorWLabels(DataGenerator):
    def __init__(self, data: List[np.ndarray], labels: List[np.ndarray], batch_size: int = 8, seq_len: int = 100,
                 step: int = 1, normalize=False,
                 normalize_function: Callable[[List, Tuple], List] = None, normalize_range: Tuple = (-1, 1),
                 test: bool = False, shuffle_data: bool = False):
        super().__init__(data, batch_size, seq_len, step, normalize, normalize_function, normalize_range, test,
                         shuffle_data)
        self.labels = labels

    def __get_sample(self):
        sample = self.data[self.current_file][self.n: self.n + self.seq_len]
        label = self.labels[self.current_file][self.n: self.n + self.seq_len]
        return sample, label

    def __getitem__(self, idx):
        self._go_to_idx(idx)
        samples, labels = list(), list()
        for i in range(self.batch_size):
            item, label = self.__get_sample()
            samples.append(item.tolist())
            labels.append(label.tolist())
            self.n += self.step
        samples = np.array(samples)
        labels = np.array(labels)
        samples = np.expand_dims(samples, axis=2)
        if not self.test:
            return samples
        else:
            return samples, labels

    def on_epoch_end(self):
        if self.shuffle_data:
            temp = list(zip(self.data, self.labels, self.num_samples, self.num_batches))
            random.shuffle(temp)
            self.data, self.labels, self.num_samples, self.num_batches = zip(*temp)
            self.accum_num_batches = list(accumulate(self.num_batches))
