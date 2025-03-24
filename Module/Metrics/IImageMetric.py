from abc import ABC, abstractmethod


class IImageMetric(ABC):
    @abstractmethod
    def Calculate(self, img1, img2):
        """Calculate the distance between two images."""
        pass
