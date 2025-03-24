from Module.Metrics import IImageMetric
import numpy as np


class CachedMetric(IImageMetric):
    def __init__(self, metric):
        """
        Args:
            metric: The base metric to use (should implement IImageMetric).
            dataset_size: Total number of images in the dataset (used for cache size).
        """
        self.metric = metric
        self.cache = {}  # Stores (idx1, idx2) -> distance

    def Calculate(self, img1, img2, idx1=None, idx2=None):
        """
        Computes or retrieves cached distance.

        Args:
            img1, img2: Image tensors.
            idx1, idx2: Optional dataset indices for caching.

        Returns:
            Cached or computed distance.
        """
        if idx1 is not None and idx2 is not None:
            key = (min(idx1, idx2), max(idx1, idx2))  # Ensure consistent key order
            if key in self.cache:
                return self.cache[key]

        # Compute distance if not cached
        distance = self.metric.Calculate(img1, img2)

        # Store in cache if indices provided
        if idx1 is not None and idx2 is not None:
            self.cache[key] = distance

        return distance
