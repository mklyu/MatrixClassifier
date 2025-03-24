import os
from typing import List, Optional
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(
        self,
        dataDir: str,
        loadBatches=["data_batch_1"],
        trimFirst: Optional[int] = None,
    ):
        assert isinstance(dataDir, str)

        self.dataDir = dataDir
        self._loadBatches = loadBatches
        self._trimFirst = trimFirst

        self._data: List = []
        self._labels: List = []

    def Load(self):
        data = []
        labels = []

        if not os.path.isdir(self.dataDir):
            raise FileNotFoundError(f"Folder {self.dataDir} not found")

        for file in os.listdir(self.dataDir):
            if file in self._loadBatches:
                batch = self._Unpickle(os.path.join(self.dataDir, file))
                data.extend(batch[b"data"])
                labels.extend(batch[b"labels"])

        # shortens torch load
        data = np.array(data)

        self._data = (
            torch.tensor(data, dtype=torch.float32).reshape(-1, 3, 32, 32) / 255.0
        )
        self._labels = torch.tensor(labels, dtype=torch.long)

        # If trimFirst is specified, trim the dataset
        if self._trimFirst is not None and isinstance(self._trimFirst, int):
            self._data = self._data[:self._trimFirst]
            self._labels = self._labels[:self._trimFirst]

    def _Unpickle(self, file):
        with open(file, "rb") as fo:
            data = pickle.load(fo, encoding="bytes")
        return data

    def __len__(self):
        return len(self._data) if self._data is not None else 0

    def __getitem__(self, idx):
        return self._data[idx], self._labels[idx]
