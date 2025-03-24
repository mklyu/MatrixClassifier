import cv2
import numpy as np
from Module.Dataset import Dataset

DATA_DIR = "./cifar-10-batches-py"


class VisualizerRoutine:
    def __init__(self, dataset, batch_size=4):
        self.dataset = dataset
        self.batch_size = batch_size

    def Run(self):
        if len(self.dataset) == 0:
            print("Dataset not loaded. Call Load() first.")
            return

        for iterImage, iterImageIndex in self.dataset:

            imageNumpy = iterImage.numpy().transpose(1, 2, 0)  # bgr to rgb
            imageNumpy = (imageNumpy * 255).astype(np.uint8)

            cv2.imshow("CIFAR-10 Image", cv2.cvtColor(imageNumpy, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

            cv2.destroyAllWindows()


if __name__ == "__main__":
    print(f"Datadir {DATA_DIR}")
    dataset = Dataset(DATA_DIR)
    print(f"Loading..")
    dataset.Load()
    print(f"Press any key to cycle images <<<")
    visualizer = VisualizerRoutine(dataset)
    visualizer.Run()
