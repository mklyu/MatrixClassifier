Please download CIFAR-10 dataset from:
https://www.cs.toronto.edu/~kriz/cifar.html

Download the python version dataset.

Extract cifar-10-batches-py from the compressed folder inside project directory

Install python3 venv 
> apt install python3-venv

Use the setup script
> sudo bash ./setup.sh

Quick exection:

> python3 KMedoidRoutine.py

Execution scripts:

> VisualizerRoutine: visualize data for a sanity check.

> VisualizeMetricRoutine: visualize the matrix norm distance used.

> CachePrecomputeRoutine: precompute the distance cache map for the dataset ,distance_cache.pkl will be exported. Only needed if dataset is trimmed to be larger than defaut.

> KMedoidRoutine: classify using K-means (medians are used, so it's actually K-medoid) and visualize medians and closest / farthest images in the feature distance space.