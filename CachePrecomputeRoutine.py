from Module.Dataset import Dataset
from Module.Metrics import MatrixNormDifference,CachedMetric

DATA_DIR = "./cifar-10-batches-py"
CACHE_FILE = "./distance_cache.pkl"


class CachePrecomputeRoutine:
    def __init__(self, dataset, metric, cache_file):
        """
        Args:
            dataset: The dataset object.
            metric: The base metric used (before caching).
            cache_file: Path to save the computed cache.
        """
        self.dataset = dataset
        self.cached_metric = CachedMetric(metric)
        self.cache_file = cache_file

    def Run(self):
        if len(self.dataset) == 0:
            print("Dataset not loaded. Call Load() first.")
            return

        print("Precomputing distances...")

        for i in range(len(self.dataset)):
            img1, _ = self.dataset[i]

            for j in range(i + 1, len(self.dataset)):  # Compute for all pairs
                img2, _ = self.dataset[j]

                # Compute and cache the distance
                self.cached_metric.Calculate(img1, img2, i, j)

        # Save cache to file
        self.cached_metric.ToPickle(self.cache_file)
        print(f"Cache saved to {self.cache_file}")


if __name__ == "__main__":
    dataset = Dataset(DATA_DIR)
    dataset.Load()

    metric = MatrixNormDifference(normType="frobenius")

    precompute_routine = CachePrecomputeRoutine(dataset, metric, CACHE_FILE)
    precompute_routine.Run()
