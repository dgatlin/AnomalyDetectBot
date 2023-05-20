import argparse
import os
import json
import pandas as pd
import numpy as np
import joblib
from container.anomaly_model import anomaly_model

path1 = "test"

if __name__ == "__main__":

    print("initializing")
    parser = argparse.ArgumentParser()
    amd = anomaly_model.AnomalyModel(path1, path1, path1)

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

    print(args)

    out = amd.adb_evaluate_model()
    print(out)
