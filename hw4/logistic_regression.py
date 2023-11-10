import numpy as np
from matplotlib import pyplot as plt
from utils import confusion
from random_generator import univar_gaussian
from argparse import ArgumentParser


class DataParam:
    mx = 0
    my = 0
    vx = 0
    vy = 0

    def __init__(self, mx: float, my: float, vx: float, vy: float) -> None:
        self.mx = mx
        self.my = my
        self.vx = vx
        self.vy = vy


def GD(weight: np.ndarray, data: np.ndarray, label: np.ndarray, lr: float):
    gradient = data.T @ (1 / (1 + np.exp(-data @ weight)) - label)
    return lr * gradient


def newton(weight: np.ndarray, data: np.ndarray, label: np.ndarray, lr: float):
    N = data.shape[0]
    D = np.zeros((N, N))
    np.fill_diagonal(D, (np.exp(-data @ weight) /
                         (1 + np.exp(-data @ weight))**2))
    hessian = data.T @ D @ data
    gradient = data.T @ (1 / (1 + np.exp(-data @ weight)) - label)
    try:
        h_inv = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        h_inv = lr * np.identity(3)
        print("newton failed")
    return h_inv @ gradient


def confusion_matrix(pred: np.ndarray, target: np.ndarray):
    print("Confusion matrix")
    pass


def logistic(N: int, D1: DataParam, D2: DataParam, max_iters=150, lr=1e-2):
    ds1 = []
    ds2 = []
    for n in range(N):
        ds1.append(
            [univar_gaussian(D1.mx, D1.vx),
             univar_gaussian(D1.my, D1.vy)])
        ds2.append(
            [univar_gaussian(D2.mx, D2.vx),
             univar_gaussian(D2.my, D2.vy)])
    ds1, ds2 = np.array(ds1), np.array(ds2)

    data = np.ones((2 * N, 3))
    data[0:N, :2] = ds1
    data[N:2 * N, :2] = ds2
    label = np.zeros((2 * N, 1))
    label[N:] = 1

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title("GT")
    plt.scatter(ds1[:, 0], ds1[:, 1], color='blue')
    plt.scatter(ds2[:, 0], ds2[:, 1], color='red')

    weight = np.zeros((3, 1))
    for it in range(max_iters):
        change = GD(weight, data, label, lr)
        weight -= change
        if np.abs(change).sum() < 1e-2:
            print(f"spent {it+1} iterations")
            break

    print("GD weight", weight)
    print()
    pred = np.squeeze(1 / (1 + np.exp(-data @ weight)))
    pred_0 = data[pred <= 0.5, :]
    pred_1 = data[pred > 0.5, :]
    cf = confusion((pred > 0.5).astype(np.int32), label.squeeze(), 0)
    print("confusion matrix")
    print(cf)
    print()
    speci = cf.iloc[1, 1]
    sensi = cf.iloc[0, 0]
    print(f"sensitivity: {sensi/N}")
    print(f"specificity: {speci/N}")
    print("\n\n")
    plt.subplot(1, 3, 2)
    plt.title("Gradient descent")
    plt.scatter(pred_0[:, 0], pred_0[:, 1], color='blue')
    plt.scatter(pred_1[:, 0], pred_1[:, 1], color='red')

    weight = np.zeros((3, 1))
    for it in range(max_iters):
        change = newton(weight, data, label, lr)
        weight -= change
        if np.abs(change).sum() < 1e-2:
            print(f"spent {it+1} iterations")
            break

    print("Newton weight", weight)
    pred = np.squeeze(1 / (1 + np.exp(-data @ weight)))
    pred_0 = data[pred <= 0.5, :]
    pred_1 = data[pred > 0.5, :]
    cf = confusion((pred > 0.5).astype(np.int32),
                   label.squeeze(),
                   0,
                   columns=["pred cluster 1", "pred cluster 2"],
                   index=["in cluster 1", "in cluster 2"])
    print("confusion matrix")
    print(cf)
    print()
    speci = cf.iloc[1, 1]
    sensi = cf.iloc[0, 0]
    print(f"sensitivity: {sensi/N}")
    print(f"specificity: {speci/N}")
    print("\n\n")
    plt.subplot(1, 3, 3)
    plt.title("Newton's method")
    plt.scatter(pred_0[:, 0], pred_0[:, 1], color='blue')
    plt.scatter(pred_1[:, 0], pred_1[:, 1], color='red')

    plt.show()


if __name__ == "__main__":
    D1_1 = DataParam(1, 1, 2, 2)
    D2_1= DataParam(10, 10, 2, 2)
    D1_2 = DataParam(1, 1, 2, 2)
    D2_2= DataParam(3, 3, 4, 4)
    # logistic(50, D1_1, D2_1)
    logistic(50, D1_2, D2_2)
