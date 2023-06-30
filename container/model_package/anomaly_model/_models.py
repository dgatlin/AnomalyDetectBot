"""
Machine learning module for Python
==================================

 The actual implementation of the machine learning algorithm within its own
 Python package. This allows me to compartmentalize the logic of the model
 with the logic needed to run it in SageMaker, and to modify and test each
 part independently. Then, the model can be reused in other environments as well.

"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from container.model_package.anomaly_model.predict_pipe import make_prediction
from container.model_package.anomaly_model.processing.data_manager import load_pipeline


class AnomalyModel:
    def __init__(self):
        self.results = None
        model_name = "anomaly_model_output_v_version.pkl"
        self.pipeline = load_pipeline(file_name=model_name)

    def predict(self, input_data: pd.DataFrame):
        self.results = make_prediction(input_data=input_data)
        return self.results

    def adb_evaluate_model(self):
        """
        Returns scores from cross-validation evaluation on the malicious/benign classifier
        """

        X_train, X_test, y_train, y_test = train_test_split(
            self.clf_X, self.clf_y, test_size=0.2, random_state=0
        )
        y_train = np.array(y_train)
        eval_cls = RandomForestClassifier(n_estimators=100, max_features=0.2)
        eval_cls.fit(X_train, y_train)

        recall = cross_val_score(eval_cls, X_train, y_train, cv=5, scoring="recall")
        precision = cross_val_score(
            eval_cls, X_train, y_train, cv=5, scoring="precision"
        )
        accuracy = cross_val_score(eval_cls, X_train, y_train, cv=5, scoring="accuracy")
        f1_score = cross_val_score(eval_cls, X_train, y_train, cv=5, scoring="f1_macro")

        return {
            "accuracy": accuracy,
            "f1": f1_score,
            "precision": precision,
            "recall": recall,
        }
