from sklearn.base import BaseEstimator
import numpy as np

from sklearn.metrics.pairwise import pairwise_distances


class DBpd (BaseEstimator):
    def __init__(self, p=0.9, d=75, verbose=1):
        self.p = p
        self.d = d
        self.verbose = verbose

    def fit(self, x):
        num_samples = x.shape[0]

        # calculate distance matrix
        dist_matrix = pairwise_distances(x, metric='euclidean')
        if self.verbose > 0:
            print('max distance: ', np.max(dist_matrix))

        dist_matrix_greater_d = dist_matrix > self.d
        sum_dist_matrix_greater_d = np.sum(dist_matrix_greater_d, axis=1)
        percent_greater_d = sum_dist_matrix_greater_d / (num_samples - 1)

        y_predict = np.ones(num_samples)
        y_predict[percent_greater_d > self.p] = -1

        return y_predict
