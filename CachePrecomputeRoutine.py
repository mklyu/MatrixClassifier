from Module.Dataset import Dataset
from Module.Metrics import MatrixNormDifference, CachedMetric
from concurrent.futures import ThreadPoolExecutor
import threading
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
        self.lock = threading.Lock()  # For thread-safe cache access

    def CalculateDistance(self, idx1, idx2):
        """Calculate distance between two images and cache the result."""
        img1, _ = self.dataset[idx1]
        img2, _ = self.dataset[idx2]

        with self.lock:  # Ensure that cache writing is thread-safe
            self.metricCache.Calculate(img1, img2, idx1, idx2)

    def Run(self):
        if len(self.dataset) == 0:
            print("Dataset not loaded. Call Load() first.")
            return

        print("Precomputing distances...")

        # Initialize a list to hold the future tasks
        futures = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
            # Use tqdm to show a progress bar over the pairs
            total_pairs = len(self.dataset) * (len(self.dataset) - 1) // 2  # Total unique pairs
            with tqdm(total=total_pairs, desc="Precomputing distances", ncols=100) as pbar:
                for iterIndexImageIn in range(len(self.dataset)):
                    for iterIndexImageOut in range(iterIndexImageIn + 1, len(self.dataset)):  # Avoid redundant pairs
                        futures.append(executor.submit(self.CalculateDistance, iterIndexImageIn, iterIndexImageOut))
                        pbar.update(1)  # Update the progress bar

                # Wait for all futures to complete
                for future in futures:
                    future.result()  # Blocks until the future is done

        # Save cache to file
        self.metricCache.ToPickle(self.cache_file)
        print(f"Cache saved to {self.cache_file}")


if __name__ == "__main__":
    dataset = Dataset(DATA_DIR)
    dataset.Load()

    metric = MatrixNormDifference(normType="frobenius")

    precompute_routine = CachePrecomputeRoutine(dataset, metric, CACHE_FILE)
    precompute_routine.Run()
