import numpy as np
from torch.utils.data import Sampler


class LevelIndexIterator:
    def __init__(self, index_list) -> None:
        self._index_list = index_list
        self._N = len(self._index_list)
        self._cur_position = 0

    def next(self):
        output_pos = self._cur_position
        self._cur_position += 1
        self._cur_position = self._cur_position % self._N
        return self._index_list[output_pos]


class ContrastiveSampler(Sampler):
    """
    It ensures that in most batches there are exactly 2 images with same level of structural noise.
    Note that it is not always true due to:
    1. If number of noise levels is less than half of the batch size, then it is not possible.
    2. In the last few batches, it may not be true.
    """
    def __init__(self, dataset, data_size, noise_levels, batch_size) -> None:
        super().__init__(dataset)
        self._dset = dataset
        self._N = data_size
        self._noise_levels = noise_levels
        self._noise_N = len(self._noise_levels)
        self._batch_N = batch_size
        assert batch_size % 2 == 0
        self.idx = None
        self.batches_levels = None

    def __iter__(self):

        level_iters = [LevelIndexIterator(self.idx.copy()) for _ in range(self._noise_N)]

        for one_batch_levels in self.batches_levels:
            batch_data_idx = []
            for level_idx in one_batch_levels:
                # two same level idx
                data_idx = level_iters[level_idx].next()
                batch_data_idx.append(self._dset.get_index(data_idx, level_idx))
                data_idx = level_iters[level_idx].next()
                batch_data_idx.append(self._dset.get_index(data_idx, level_idx))
            yield batch_data_idx

    def set_epoch(self, epoch):
        self.batches_levels = []
        for _ in range(int(np.ceil(self._N / self._batch_N))):
            if self._noise_N >= self._batch_N / 2:
                levels = np.random.choice(np.arange(self._noise_N), size=self._batch_N // 2, replace=False)
            else:
                levels = np.random.choice(np.arange(self._noise_N), size=self._batch_N // 2, replace=True)

            self.batches_levels.append(levels)

        self.idx = np.arange(self._batch_N)
        np.random.shuffle(self.idx)
