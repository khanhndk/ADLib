from sklearn.base import BaseEstimator
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances


class LOFDetector (BaseEstimator):
    def __init__(self, k_nearest=3, top_anomaly=100, verbose=0):
        self.k_nearest = k_nearest
        self.top_anomaly = top_anomaly
        self.verbose = verbose

    def fit(self, x):
        num_samples = x.shape[0]

        # find k-NN
        nbrs = NearestNeighbors(n_neighbors=self.k_nearest + 1, algorithm='ball_tree').fit(x)
        dist_knn, indices = nbrs.kneighbors(x)

        # calculate distance matrix
        dist_matrix = pairwise_distances(x, metric='euclidean')

        # calculate N_k
        n_k = np.zeros(num_samples, dtype=int)
        n_k_idx = []
        for ni in range(num_samples):
            n_k_idx.append([])
            for nj in range(num_samples):
                if dist_matrix[ni, nj] <= dist_knn[ni, self.k_nearest]:
                    n_k[ni] += 1
                    n_k_idx[ni].append(nj)

        lrd_k = np.zeros(num_samples)
        for ni in range(num_samples):
            for nj in n_k_idx[ni]:
                lrd_k[ni] += np.maximum(dist_matrix[ni, nj], dist_knn[ni, self.k_nearest])
            lrd_k[ni] = n_k[ni] / lrd_k[ni]

        lof_k = np.zeros(num_samples)
        for ni in range(num_samples):
            tmp_sum = 0
            for nj in n_k_idx[ni]:
                tmp_sum += lrd_k[nj]
            lof_k[ni] = tmp_sum / (lrd_k[ni] * n_k[ni])

        idx_anomaly = lof_k.argsort()[-self.top_anomaly:][::-1]

        y_predict = np.ones(num_samples)
        y_predict[idx_anomaly] = -1

        return y_predict
