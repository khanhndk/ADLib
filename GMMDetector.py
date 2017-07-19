import numpy as np
from sklearn import mixture


class GMMDetector:
    def __init__(self, num_mixtures=5, percent_abnormal=0.005, verbose=1):
        self.num_mixtures = num_mixtures
        self.percent_abnormal = percent_abnormal
        self.verbose = verbose

        self.x_train = None
        self.gmm = None

    def fit(self, x_train):
        self.x_train = x_train

        # Init Gaussian Mixture Model with 5 components
        self.gmm = mixture.GaussianMixture(self.num_mixtures)
        self.gmm.fit(self.x_train)

    def predict(self, x_predict):
        # Estimate log probability
        log_prob = self.gmm.score_samples(x_predict)
        log_prob = log_prob * (-1)
        log_prob = log_prob + abs(np.min(log_prob)) + 0.5
        residual_signal = log_prob

        a = np.sort(residual_signal)
        threshold = a[int((1 - self.percent_abnormal) * residual_signal.size)]

        num_samples = x_predict.shape[0]
        y_predict = np.ones(num_samples)
        y_predict[residual_signal > threshold] = -1
        return y_predict
