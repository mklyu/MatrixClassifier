import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Module.Dataset import Dataset
from Module.Metrics import IImageMetric, MatrixNormDifference

from Util.OpencvPyqt5Fix import FUNC_FIX_OPENCV_PYQT5_XCB

FUNC_FIX_OPENCV_PYQT5_XCB()

DATA_DIR = "./cifar-10-batches-py"


class VisualizeMetricRoutine:
    def __init__(self, dataset, metric, num_samples=100):
        """
        Args:
            dataset: The dataset object.
            metric: The metric used to calculate distances (should implement IImageMetric).
            num_samples: The number of random samples to calculate distances to.
        """
        self.dataset = dataset
        self.metric: IImageMetric = metric
        self.numSamples = num_samples

    def Run(self):
        if len(self.dataset) == 0:
            print("Dataset not loaded. Call Load() first.")
            return

        # Select a random image from the dataset
        randomIndex = random.randint(0, len(self.dataset) - 1)
        randomImage, _ = self.dataset[randomIndex]

        # Calculate the distances to other random images
        distances = []
        randomIndices = random.sample(range(len(self.dataset)), self.numSamples)

        for idx in randomIndices:
            img, _ = self.dataset[idx]
            distance = self.metric.Calculate(randomImage, img)
            distances.append((distance, img))  # Store distance and image

        # Sort by distance
        distances.sort(key=lambda x: x[0])
        closestImage = distances[0][1]
        farthestImage = distances[-1][1]
        distanceValues = [d[0] for d in distances]

        # Plot the graph of distances + images
        self.PlotDistances(distanceValues, randomImage, closestImage, farthestImage)

    def PlotDistances(self, distances, refImage, closestImage, farthestImage):
        """Plot a histogram of distances and show closest/farthest images."""
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))

        # Histogram
        axes[0].hist(distances, bins=20, color="blue", edgecolor="black")
        axes[0].set_title("Histogram of Image Distances")
        axes[0].set_xlabel("Distance")
        axes[0].set_ylabel("Images")
        axes[0].grid(True)

        # Display Reference, Closest, and Farthest images
        for i, (img, title) in enumerate(zip([refImage, closestImage, farthestImage], 
                                             ["Reference", "Closest", "Farthest"])):
            img_np = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            axes[i+1].imshow(img_bgr)
            axes[i+1].set_title(title)
            axes[i+1].axis("off")

        plt.show()


if __name__ == "__main__":
    dataset = Dataset(DATA_DIR)
    dataset.Load()

    metric = MatrixNormDifference(normType="frobenius")

    # Initialize and run the VisualizerRoutine
    visualizer = VisualizeMetricRoutine(dataset, metric)
    visualizer.Run()
