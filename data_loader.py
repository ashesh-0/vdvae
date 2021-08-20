import numpy as np


class MNISTNoisyLoader:
    """
    Different versions of MNIST are loader. Each version corresponds to a different noise level. Except for the noise
     level, everything else is same. To use it to train the model, one needs a custom batch_sampler.
    """
    def __init__(self, directory_dict: dict) -> None:
        dir_tuple = [(nlevel, dir) for nlevel, dir in directory_dict.items()]
        dir_tuple = sorted(dir_tuple, key=lambda x: x[0])
        self.noise_levels = [x[0] for x in dir_tuple]
        self._dir_list = [x[1] for x in dir_tuple]

        print('[{self.__class__.__name__}] Noise levels:', self.noise_levels)
        self.N = None
        self._noise_level_count = len(self._dir_list)
        self._all_data = self.load()

    def load_single(directory):
        pass

    def load(self):
        data = {}
        for noise_level, directory in self._dir_list:
            data[noise_level] = self.load_single(directory)

        sz = data[noise_level].shape[0]
        for nlevel in data:
            assert data[nlevel].shape[0] == sz
        self.N = sz
        return data

    def __getitem__(self, index):
        noise_level = index // self.N
        new_index = index - self.N * noise_level
        return self._all_data[noise_level][new_index], np.array([self._dir_list[noise_level][0]])

    def __len__(self):
        return self.N * self._noise_level_count
