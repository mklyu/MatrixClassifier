import pickle
from .IImageMetric import IImageMetric


class CachedMetric(IImageMetric):
    def __init__(self, metric):
        """
        Args:
            metric: The base metric to use (should implement IImageMetric).
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
        # Ensure that the indices are order-independent for the cache
        if idx1 is not None and idx2 is not None:
            # Using sorted indices to ensure the key order does not matter
            key = tuple(sorted((idx1, idx2)))  # Same key for both (idx1, idx2) and (idx2, idx1)
            if key in self.cache:
                return self.cache[key]

        # Compute distance if not cached
        distance = self.metric.Calculate(img1, img2)

        # Store in cache if indices provided
        if idx1 is not None and idx2 is not None:
            self.cache[key] = distance

        return distance

    def ToPickle(self, file_path):
        """Saves the cached distances to a file."""
        with open(file_path, "wb") as f:
            pickle.dump(self.cache, f)

    def FromPickle(self, file_path):
        """Loads cached distances from a file."""
        with open(file_path, "rb") as f:
            self.cache = pickle.load(f)
