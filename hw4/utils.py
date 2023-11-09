import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def print_confusion(pred: np.ndarray, target: np.ndarray):
    print("Confusion matrix: ")
    TP = pred == target and target == 0
    TN = pred == target and target == 1
    FP = pred != target and target == 1
    FN = pred != target and target == 0
    df = pd.DataFrame(
        [np.sum(TP), np.sum(FN),
         np.sum(FP), np.sum(TN)],
        columns=["predicted 1", "predicted 2"],
        index=["in cluster 1", "in cluster 2"])
    print(df)
    

def specitivity(pred:np.ndarray, target:np.ndarray):
    pass

def sensitivity(pred:np.ndarray, target:np.ndarray):
    pass
