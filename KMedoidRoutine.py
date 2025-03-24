import random
from Module.Dataset import Dataset
from Module.Metrics import IImageMetric, MatrixNormDifference, CachedMetric

# Global settings
K = 3  # Number of clusters
CLASSIF_ITERS = 10  # Number of iterations for clustering
DATA_DIR = "./cifar-10-batches-py"
CACHE_FILE = "./distance_cache.pkl"


class KMedoidRoutine:
    def __init__(self, dataset, metric):
        """
        Args:
            dataset: The dataset object.
            metric: The metric used to calculate distances (should implement IImageMetric).
        """
        self.dataset = dataset
        self.metric: IImageMetric = metric
        self.data = [img for img, _ in self.dataset]  # Load all images
        self.numSamples = len(self.data)

        self.medoids = random.sample(self.data, K)  # Pick initial K medoids

    def Run(self):
        if len(self.dataset) == 0:
            print("Dataset not loaded. Call Load() first.")
            return

        for classifierIter in range(CLASSIF_ITERS):
            print(f"Iteration {classifierIter+1}/{CLASSIF_ITERS}")

            # Assign each image to the closest medoid
            clusters = {medoid: [] for medoid in self.medoids}

            for iterImage in self.data:
                closestMedoid = min(self.medoids, key=lambda medoid: self.metric.Calculate(iterImage, medoid))
                clusters[closestMedoid].append(iterImage)

            # Update medoids to be the most central image in each cluster
            newMedoids = []
            for iterMedoid, iterCluster in clusters.items():
                newMedoid = min(iterCluster, key=lambda img: sum(self.metric.Calculate(img, other) for other in iterCluster))
                newMedoids.append(newMedoid)

            # Stop if medoids did not change
            if set(map(id, newMedoids)) == set(map(id, self.medoids)):
                print("Converged early")
                break

            self.medoids = newMedoids

        print("Clustering complete")

    def GetClusters(self):
        """Returns the final cluster assignments."""
        clusters = {medoid: [] for medoid in self.medoids}
        for img in self.data:
            closest_medoid = min(self.medoids, key=lambda medoid: self.metric.Calculate(img, medoid))
            clusters[closest_medoid].append(img)
        return clusters


if __name__ == "__main__":
    dataset = Dataset(DATA_DIR,trimFirst=100)
    dataset.Load()

    metric = MatrixNormDifference(normType="frobenius")
    metricCache = CachedMetric(metric)
    metricCache.FromPickle(CACHE_FILE)

    # Run K-Medoid Clustering
    kmedoid = KMedoidRoutine(dataset, metricCache)
    kmedoid.Run()

    clusters = kmedoid.GetClusters()
    print(f"Generated {len(clusters)} clusters.")
