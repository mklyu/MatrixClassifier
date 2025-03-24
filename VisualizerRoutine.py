import cv2
import numpy as np
import matplotlib.pyplot as plt
from Dataset import Dataset

DATA_DIR = "./cifar-10-batches-py"


class VisualizerRoutine:
    def __init__(self, dataset, batch_size=4):
        self.dataset = dataset
        self.batch_size = batch_size

    def Run(self):
        if self.dataset.__len__() == 0:
            print("Dataset not loaded. Call Load() first.")
            return

        for i in range(0, len(self.dataset), self.batch_size):
            images = self.dataset.data[i:i+self.batch_size].numpy().transpose(0, 2, 3, 1)  # Convert to (batch, H, W, C)
            images = (images * 255).astype(np.uint8)  # Convert to uint8 for OpenCV
            
            fig, axes = plt.subplots(1, len(images), figsize=(10, 5))
            for img, ax in zip(images, axes):
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                ax.axis("off")
            
            plt.show()
            input("Press Enter to continue...")


if __name__ == "__main__":
    dataset = Dataset(DATA_DIR)
    dataset.Load()
    visualizer = VisualizerRoutine(dataset)
    visualizer.Run()
