from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
import numpy as np


class DBkN(BaseEstimator):
    def __init__(self, k_nearest=10, top_anomaly=100, verbose=1):
        self.k_nearest = k_nearest
        self.top_anomaly = top_anomaly
        self.verbose = verbose

    def fit(self, x):
        num_samples = x.shape[0]

        # find k-NN
        nbrs = NearestNeighbors(n_neighbors=self.k_nearest + 1, algorithm='ball_tree').fit(x)
        distances, indices = nbrs.kneighbors(x)

        avg_distances = np.average(distances, axis=1)

        idx_anomaly = avg_distances.argsort()[-self.top_anomaly:][::-1]

        y_predict = np.ones(num_samples)
        y_predict[idx_anomaly] = -1

        return y_predict
