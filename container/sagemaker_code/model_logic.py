# https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html?swcfpc=1

import argparse
import os
import pandas as pd
import numpy as np
from container.anomaly_model import anomaly_model

path1 = "/Users/dariusmac/PycharmProjects/AnomalyDetectBot/data/MalwareArtifacts.csv"

if __name__ == "__main__":

    print("initializing")
    parser = argparse.ArgumentParser()
    adb = anomaly_model.AnomalyModel()

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--train-file", type=str)
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--test-file", type=str, default=None)

    print("reading arguments")
    args, _ = parser.parse_known_args()

    print("reading training data")
    # assume there's no headers and the target is the last column
    # data = np.loadtxt(os.path.join(args.train, args.train_file), delimiter=',')
    # X = data[:, :-1]
    # y = data[:, -1]

    data = np.array(pd.read_csv(path1, header=None))
    X = data[:, :-1]
    y = data[:, -1]

    print("fitting model")
    adb.build_models(X, y)

    print("evaluating model")
    results = adb.adb_evaluate_model(X, y)
    print(results)

    print("saving model")
    # path = os.path.join(args.model_dir, "model.joblib")
    # print(f"saving to {path}")
    # joblib.dump(adb, path)

    def predict_fn(input_object, model):
        return adb.predict(input_object)
