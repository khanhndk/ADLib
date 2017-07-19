from DBpd import DBpd
from DBkN import DBkN
from LOFDetector import LOFDetector
from PCADetector import PCADetector
from GMMDetector import GMMDetector

import pandas as pd

from sklearn import metrics

# load data
df = pd.read_csv('emnist.digits_letters.small.csv', index_col=0)
data_array = df.as_matrix()
x = data_array[:, 1:]
y_true = data_array[:, 0]  # load this for testing

num_samples = x.shape[0]
print('Number of samples:', num_samples)

# run DBpd
est_DBpd = DBpd(d=74.0, p=0.007)
y_predict = est_DBpd.fit(x)
print('Classification results for DBpd:')
print(metrics.classification_report(y_true, y_predict))

# run DBkN
est_DBkN = DBkN(k_nearest=10, top_anomaly=100)
y_predict = est_DBkN.fit(x)
print('Classification results for DBkN:')
print(metrics.classification_report(y_true, y_predict))

# run LOFDetector
est_LOFDetector = LOFDetector(k_nearest=3, top_anomaly=100)
y_predict = est_LOFDetector.fit(x)
print('Classification results for LOFDetector:')
print(metrics.classification_report(y_true, y_predict))

# run PCADetector
est_PCADetector = PCADetector(keep_info=0.999, find_threshold_type='percentile', percent_normal=90)
est_PCADetector.fit(x[y_true > 0])  # only input normal data
y_predict = est_PCADetector.predict(x)
print('Classification results for PCADetector:')
print(metrics.classification_report(y_true, y_predict))

# run GMMDetector
est_GMMDetector = GMMDetector(num_mixtures=5, percent_abnormal=0.005,)
est_GMMDetector.fit(x[y_true > 0])  # only input normal data
y_predict = est_GMMDetector.predict(x)
print('Classification results for PCADetector:')
print(metrics.classification_report(y_true, y_predict))
