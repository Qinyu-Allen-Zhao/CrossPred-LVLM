import abc
import numpy as np

class BaseDataset(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def get_data(self, chunk_id):
        pass

    def sample(self, num_samples):
        data_id = np.arange(len(self))

        if num_samples is not None:
            np.random.seed(0)
            data_id = np.random.choice(data_id, num_samples, replace=False)

        return data_id
