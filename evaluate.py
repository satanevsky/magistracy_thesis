import collections
import os
from os import path as os_path
import pandas as pd
from sklearn import metrics
from sklearn import feature_selection
from sklearn import base as sklearn_base
from sklearn import model_selection

DATA_DIR = 'data'

_TEXT = 'text'

def get_data():
    data_frames = list()
    for el in os.listdir(DATA_DIR):
        el = os_path.join(DATA_DIR, el)
        df = pd.read_csv(el)[['text', 'sentiment']]
        df.sentiment = df.sentiment.astype(int)
        data_frames.append(df)

    return pd.concat(data_frames).groupby('text').max().reset_index()


def mean(x):
    return sum(x) / len(x)


class Evaluator(object):
    def __init__(self):
        self._data = get_data()

    def evaluate(self, model):
        model_metrics = collections.defaultdict(list)
        for train_ind, test_ind in model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(self._data.text, self._data.sentiment):
            X_train = self._data.text[train_ind]
            y_train = self._data.sentiment[train_ind]
            X_test = self._data.text[test_ind]
            y_test = self._data.sentiment[test_ind]
            predictions = sklearn_base.clone(model).fit(X_train, y_train).predict_proba(X_test)
            model_metrics['auc'].append(
                metrics.roc_auc_score(y_true=y_test, y_score=predictions[:,1])
            )
        return model_metrics
    
