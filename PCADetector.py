from sklearn.base import BaseEstimator
import numpy as np
import scipy


class PCADetector (BaseEstimator):
    def __init__(
            self,
            keep_info=0.999,
            find_threshold_type='percentile',
            alpha=0.3,  # use for find_threshold_type = 'keepinfo'
            percent_normal=90,  # use for find_threshold_type = 'percentile'
            verbose=0
    ):
        self.keep_info = keep_info
        self.find_threshold_type = find_threshold_type
        self.alpha = alpha
        self.percent_normal = percent_normal
        self.verbose = verbose

        self.num_principal_components = 0
        self.s = None
        self.u2 = None
        self.residual = None
        if find_threshold_type == 'keepinfo':
            self.find_threshold = self.find_threshold_keepinfo
        elif find_threshold_type == 'percentile':
            self.find_threshold = self.find_threshold_percentile

    def find_threshold_percentile(self):
        return np.percentile(self.residual, self.percent_normal)

    def find_threshold_keepinfo(self):
        c_alpha = scipy.stats.norm.ppf(1 - self.alpha)
        phi1 = np.sum(self.s[self.num_principal_components + 1:])
        phi2 = np.sum(self.s[self.num_principal_components + 1:] ** 2)
        phi3 = np.sum(self.s[self.num_principal_components + 1:] ** 3)
        h0 = 1 - (2 * phi1 * phi3) / (3 * phi2 * phi2)
        threshold = \
            phi1 * (
                       c_alpha * np.sqrt(2 * phi2 * h0 * h0) / phi1
                       + 1 + (phi2 * h0 * (h0 - 1)) / (phi1 * phi1)
                   ) ** (1 / h0)
        return threshold

    def fit(self, x):
        """
        :param x: input only normal data
        :return:
        """
        cov_mat = np.cov(x.T)  # only consider normal data

        [u, self.s, _] = np.linalg.svd(cov_mat)

        cs = self.s.cumsum()
        self.num_principal_components = \
            int(np.where(cs >= cs[-1] * self.keep_info)[0][0] + 1)
        if self.verbose > 0:
            print('Number of principal components:', self.num_principal_components)

        self.u2 = u[:, self.num_principal_components:]  # residual subspace

    def predict(self, x):
        num_samples = x.shape[0]

        # project onto residual subspace
        self.residual = np.power(x.dot(self.u2.dot(self.u2.T)), 2).sum(axis=1)

        # automatically compute threshold
        threshold = self.find_threshold()
        if self.verbose > 0:
            print("Auto threshold = {}".format(threshold))

        y_predict = np.ones(num_samples)
        y_predict[self.residual > threshold] = -1

        return y_predict
