import numpy as np
from skimage.transform import resize


class MNISTNoisyLoader:
    """
    Different versions of MNIST are loader. Each version corresponds to a different noise level. Except for the noise
     level, everything else is same. To use it to train the model, one needs a custom batch_sampler.
    """
    def __init__(self, fpath_dict: dict) -> None:
        fpath_noise_tuple = [(nlevel, dir) for nlevel, dir in fpath_dict.items()]
        fpath_noise_tuple = sorted(fpath_noise_tuple, key=lambda x: x[0])
        self.noise_levels = [x[0] for x in fpath_noise_tuple]
        self._fpath_list = [x[1] for x in fpath_noise_tuple]

        print(f'[{self.__class__.__name__}] Noise levels:', self.noise_levels)
        self.N = None
        self._noise_level_count = len(self._fpath_list)
        self._all_data = self.load()

    def load(self):
        data = {}
        for noise_index, fpath in enumerate(self._fpath_list):
            noisy_data = np.load(fpath)
            noisy_data = resize(noisy_data, (noisy_data.shape[0], 32, 32, 1))
            data[noise_index] = np.repeat(noisy_data, 3, axis=3)
        sz = data[noise_index].shape[0]
        for nlevel in data:
            assert data[nlevel].shape[0] == sz
        self.N = sz
        return data

    def __getitem__(self, index):
        noise_index = index // self.N
        new_index = index - self.N * noise_index
        return self._all_data[noise_index][new_index], np.array([self.noise_levels[noise_index]])

    def get_index(self, index, noise_level_index):
        return self.N * noise_level_index + index

    def __len__(self):
        return self.N * self._noise_level_count
