from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def confusion(pred: np.ndarray,
              target: np.ndarray,
              label: int,
              columns: List = ["pred true", "pred false"],
              index: List = ["true", "false"]) -> pd.DataFrame:
    TP = (pred == label).astype(np.int32) * (target == label).astype(np.int32)
    TN = (pred != label).astype(np.int32) * (target != label).astype(np.int32)
    FP = (pred == label).astype(np.int32) * (pred != label).astype(np.int32)
    FN = (pred != label).astype(np.int32) * (target == label).astype(np.int32)
    df = pd.DataFrame(
        [[np.sum(TP), np.sum(FN)], [np.sum(FP), np.sum(TN)]], columns=columns, index=index)

    return df
