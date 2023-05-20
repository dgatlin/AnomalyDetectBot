# ********************************************************************************
# The actual implementation of the machine learning algorithm within its own
# Python package. This allows me to compartmentalize the logic of the model
# with the logic needed to run it in SageMaker, and to modify and test each
# part independently. Then, the model can be reused in other environments as well.
# *********************************************************************************

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

prefix = "../ml/"


class AnomalyModel:
    def __init__(self, benign_path, anomaly_path, model_path):
        self.clear_state()
        self.set_model_paths(benign_path, anomaly_path, model_path)
        self.set_data_dfs()

    def clear_state(self):
        """
        Resets object's state to clear out all model internals created after loading state from disk
        """
        self.cls = None
        self.modeldata = None
        self.features = {}

    def set_model_paths(self, benign_path, anomaly_path, model_path):
        """
        :param benign_path:
        :param anomaly_path:
        :param model_path:
        :return:
        """
        self.benign_path = benign_path
        self.anomaly_path = anomaly_path
        self.model_path = model_path

    def set_data_dfs(self):
        benign = pd.read_csv(self.benign_path)
        anomaly = pd.read_csv(self.anomaly_path)

        self.df = pd.concat([benign, anomaly])
        print(self.df)
        self.clf_X = self.df.drop(["label"], axis=1)
        self.clf_y = np.array(self.df["label"])

    def build_models(self):
        """
        This function builds the models based on
        the classifier matrix and labels.
        :return:
        """
        self.cls = RandomForestClassifier(n_estimators=100, max_features=0.2)

        # build classifier
        self.cls.fit(self.clf_X, self.clf_y)

        return self.cls

    def adb_evaluate_model(self):
        """
        Returns scores from cross validation evaluation on the malicious / benign classifier
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

    # def adb_save_mode(self):

    def adb_predict(self, X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("sample_input must be pandas DataFrame")
        if len(X) <= 0:
            return pd.DataFrame()

        if X is not None and len(X) > 0:
            complete_result = self.cls.predict(X)
            return complete_result
        else:
            raise ValueError("Unexpected error occurred.")
