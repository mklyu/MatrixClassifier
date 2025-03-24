from .IImageMetric import IImageMetric

import numpy as np


class MatrixNormDifference(IImageMetric):
    def __init__(self, normType: str = "frobenius"):
        """
        Args:
            norm_type (str): The type of norm to use. "frobenius" | "l2"
        """
        self.normType: str = normType

    def Calculate(self, img1, img2):
        """
        Calculate the norm difference between two images.

        Args:
            img1 (np.ndarray): First image (H, W, C).
            img2 (np.ndarray): Second image (H, W, C).

        Returns:
            float: Calculated distance between the images.
        """
        # Ensure both images are numpy arrays
        img1 = np.array(img1)
        img2 = np.array(img2)

        if img1.shape != img2.shape:
            raise ValueError("Images mshape mistmatch")

        # Calculate the difference between the images
        diff = img1 - img2

        # Compute norm based on desired type
        if self.normType == "frobenius":
            # Frobenius norm (matrix norm)
            return np.linalg.norm(diff)
        elif self.normType == "l2":
            # L2 norm (Euclidean distance)
            return np.sqrt(np.sum(diff**2))
        else:
            raise ValueError(f"Unsupported norm type: {self.normType}")
