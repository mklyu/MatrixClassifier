import os
import random
from Module.Dataset import Dataset
from Module.Metrics import MatrixNormDifference
from Module.Metrics import CachedMetric

DATA_DIR = "./cifar-10-batches-py"
CACHE_FILE = "./distance_cache.pkl"
NUM_SAMPLES = 500  # Adjust for performance


class CachePrecomputeRoutine:
    def __init__(self, dataset, metric, cache_file, num_samples):
        """
        Args:
            dataset: The dataset object.
            metric: The base metric used (before caching).
            cache_file: Path to save the computed cache.
            num_samples: Number of image pairs to compute (reduce for faster processing).
        """
        self.dataset = dataset
        self.cached_metric = CachedMetric(metric)
        self.cache_file = cache_file
        self.num_samples = num_samples

    def Run(self):
        if len(self.dataset) == 0:
            print("Dataset not loaded. Call Load() first.")
            return

        print("Precomputing distances...")

        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        for i in range(min(self.num_samples, len(indices))):
            idx1 = indices[i]
            img1, _ = self.dataset[idx1]

            for j in range(i + 1, min(i + 20, len(indices))):  # Compute for nearby indices
                idx2 = indices[j]
                img2, _ = self.dataset[idx2]

                # Compute and cache the distance
                self.cached_metric.Calculate(img1, img2, idx1, idx2)

        # Save cache to file
        self.cached_metric.ToPickle(self.cache_file)
        print(f"Cache saved to {self.cache_file}")


if __name__ == "__main__":
    dataset = Dataset(DATA_DIR)
    dataset.Load()

    metric = MatrixNormDifference(normType="frobenius")

    precompute_routine = CachePrecomputeRoutine(dataset, metric, CACHE_FILE, NUM_SAMPLES)
    precompute_routine.Run()
