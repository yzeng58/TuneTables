import unittest

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tunetables.scripts.transformer_prediction_interface import TuneTablesClassifier


class TestLoadModuleOnlyInference(unittest.TestCase):
    def test_main(self):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        print("Size of X_train in test module:", X_train.shape)
        print("Size of X_test in test module:", X_test.shape)
        clf = TuneTablesClassifier()    
        clf.fit(X_train,y_train)
        # y_test = np.random.permutation(y_test)
        y_eval = clf.predict(X_test, y_test)
        print("y test:", y_test)
        print("y eval:", y_eval)
        print("Size of y_test:", y_test.shape)
        print("Size of y_eval:", y_eval.shape)
        accuracy = accuracy_score(y_test, y_eval)
        print("Accuracy:", np.round(accuracy, 2))