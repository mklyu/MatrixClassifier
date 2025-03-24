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
        Computes the distance between two images (no caching).

        Args:
            img1, img2: Image tensors.
            idx1, idx2: Optional dataset indices for caching.

        Returns:
            Computed distance.
        """
        # Compute distance without using cache
        return self.metric.Calculate(img1, img2)

    def ComputeAndCache(self, img1, img2, idx1, idx2):
        """
        Computes the distance and caches it.

        Args:
            img1, img2: Image tensors.
            idx1, idx2: Indices for caching the distance.

        Returns:
            The computed distance after caching.
        """
        distance = self.Calculate(img1, img2, idx1, idx2)  # Compute the distance
        self.Cache(distance, idx1, idx2)  # Cache the computed distance
        return distance

    def Cache(self, distance, idx1, idx2):
        """
        Directly caches the given distance with the provided indices.

        Args:
            distance: The computed distance between two images.
            idx1, idx2: Indices for caching the distance.
        """
        # Ensure the cache key order is independent of the indices' order
        key = tuple(sorted((idx1, idx2)))
        self.cache[key] = distance

    def ToPickle(self, file_path):
        """Saves the cached distances to a file."""
        with open(file_path, "wb") as f:
            pickle.dump(self.cache, f)

    def FromPickle(self, file_path):
        """Loads cached distances from a file."""
        with open(file_path, "rb") as f:
            self.cache = pickle.load(f)
