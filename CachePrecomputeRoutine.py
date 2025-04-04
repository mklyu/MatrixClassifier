from Module.Dataset import Dataset
from Module.Metrics import MatrixNormDifference, CachedMetric
from tqdm import tqdm

DATA_DIR = "./cifar-10-batches-py"
CACHE_FILE = "./distance_cache.pkl"
WORKER_COUNT = 12


class CachePrecomputeRoutine:
    def __init__(self, dataset, metric, cache_file):
        """
        Args:
            dataset: The dataset object.
            metric: The base metric used (before caching).
            cache_file: Path to save the computed cache.
        """
        self.dataset = dataset
        self.metricCache = CachedMetric(metric)
        self.cache_file = cache_file

    def CalculateDistance(self, idx1, idx2):
        """Calculate distance between two images and cache the result."""
        img1, _ = self.dataset[idx1]
        img2, _ = self.dataset[idx2]

        self.metricCache.Calculate(img1, img2, idx1, idx2)

    def Run(self):
        if len(self.dataset) == 0:
            print("Dataset not loaded. Call Load() first.")
            return

        print("Precomputing distances...")

        for iterIndexImageIn in tqdm(range(len(self.dataset))):
            for iterIndexImageOut in range(
                iterIndexImageIn + 1, len(self.dataset)
            ):  # Avoid redundant pairs
                self.CalculateDistance(iterIndexImageIn, iterIndexImageOut)


        self.metricCache.ToPickle(self.cache_file)
        print(f"Cache saved to {self.cache_file}")


if __name__ == "__main__":
    dataset = Dataset(DATA_DIR, trimFirst=1000)
    dataset.Load()

    metric = MatrixNormDifference(normType="frobenius")

    precompute_routine = CachePrecomputeRoutine(dataset, metric, CACHE_FILE)
    precompute_routine.Run()
