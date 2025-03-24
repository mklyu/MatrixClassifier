import random
import torch
import matplotlib.pyplot as plt
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
            print(f"Iteration {classifierIter + 1}/{CLASSIF_ITERS}")

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

    def VisualizeClusters(self):
        """Visualize medoids and their closest and farthest images."""
        clusters = self.GetClusters()
        fig, axes = plt.subplots(K, 7, figsize=(14, 12))  # 7 columns: medoid + 3 closest + 3 farthest
        
        for i, (medoid, images) in enumerate(clusters.items()):
            # Plot the medoid
            ax = axes[i, 0]
            ax.imshow(medoid.permute(1, 2, 0).numpy())  # Convert to HWC format for matplotlib
            ax.set_title(f"Medoid {i + 1}")
            ax.axis('off')
            
            # Plot the 3 closest images
            closest_images = sorted(images, key=lambda img: self.metric.Calculate(medoid, img))[:3]
            for j, img in enumerate(closest_images):
                ax = axes[i, j + 1]
                ax.imshow(img.permute(1, 2, 0).numpy())
                ax.axis('off')
            
            # Plot the 3 farthest images
            farthest_images = sorted(images, key=lambda img: self.metric.Calculate(medoid, img), reverse=True)[:3]
            for j, img in enumerate(farthest_images):
                ax = axes[i, j + 4]
                ax.imshow(img.permute(1, 2, 0).numpy())
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    def PrintClusterSizes(self):
        """Print the size and relative size of each cluster."""
        clusters = self.GetClusters()
        total_samples = len(self.data)

        for i, (medoid, cluster) in enumerate(clusters.items()):
            cluster_size = len(cluster)
            relative_size = cluster_size / total_samples * 100  # Percentage
            print(f"Cluster {i + 1} (Medoid {i + 1}) size: {cluster_size} ({relative_size:.2f}%)")


if __name__ == "__main__":
    dataset = Dataset(DATA_DIR, trimFirst=1000)
    dataset.Load()

    metric = MatrixNormDifference(normType="frobenius")
    metricCache = CachedMetric(metric)
    metricCache.FromPickle(CACHE_FILE)

    # Run K-Medoid Clustering
    kmedoid = KMedoidRoutine(dataset, metricCache)
    kmedoid.Run()

    # Visualize the medoids and their closest and farthest images
    kmedoid.VisualizeClusters()

    # Print cluster sizes and their relative sizes
    kmedoid.PrintClusterSizes()
