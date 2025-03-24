import random
from Module.Dataset import Dataset
from Module.Metrics import IImageMetric, MatrixNormDifference

# Global settings
K = 3  # Number of clusters
CLASSIF_ITERS = 10  # Number of iterations for clustering
DATA_DIR = "./cifar-10-batches-py"


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

        for iteration in range(CLASSIF_ITERS):
            print(f"Iteration {iteration+1}/{CLASSIF_ITERS}")

            # Assign each image to the closest medoid
            clusters = {medoid: [] for medoid in self.medoids}
            for img in self.data:
                closest_medoid = min(self.medoids, key=lambda medoid: self.metric.Calculate(img, medoid))
                clusters[closest_medoid].append(img)

            # Update medoids to be the most central image in each cluster
            new_medoids = []
            for medoid, cluster in clusters.items():
                new_medoid = min(cluster, key=lambda img: sum(self.metric.Calculate(img, other) for other in cluster))
                new_medoids.append(new_medoid)

            # Stop if medoids did not change
            if set(map(id, new_medoids)) == set(map(id, self.medoids)):
                print("Converged early!")
                break

            self.medoids = new_medoids

        print("Clustering complete!")

    def GetClusters(self):
        """Returns the final cluster assignments."""
        clusters = {medoid: [] for medoid in self.medoids}
        for img in self.data:
            closest_medoid = min(self.medoids, key=lambda medoid: self.metric.Calculate(img, medoid))
            clusters[closest_medoid].append(img)
        return clusters


if __name__ == "__main__":
    dataset = Dataset(DATA_DIR)
    dataset.Load()

    metric = MatrixNormDifference(normType="frobenius")

    # Run K-Medoid Clustering
    kmedoid = KMedoidRoutine(dataset, metric)
    kmedoid.Run()

    clusters = kmedoid.GetClusters()
    print(f"Generated {len(clusters)} clusters.")
