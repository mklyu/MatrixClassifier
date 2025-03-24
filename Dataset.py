import os
from typing import List
import pickle

import torch
from torch.utils.data import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(self, filedir: str):
        assert isinstance(filedir, str)

        self.filedir = filedir

        self._data: List = []
        self._labels: List = []

    def Load(self):
        data = []
        labels = []

        for file in os.listdir(self.filedir):
            if file.startswith("data_batch") or file.startswith("test_batch"):
                batch = self._Unpickle(os.path.join(self.filedir, file))
                data.extend(batch[b"data"])
                labels.extend(batch[b"labels"])

        self._data = (
            torch.tensor(data, dtype=torch.float32).reshape(-1, 3, 32, 32) / 255.0
        )
        self._labels = torch.tensor(labels, dtype=torch.long)

    def _Unpickle(file):
        with open(file, "rb") as fo:
            data = pickle.load(fo, encoding="bytes")
        return data

    def __len__(self):
        return len(self._data) if self._data is not None else 0

    def __getitem__(self, idx):
        return self._data[idx], self._labels[idx]
