from typing import Literal
import numpy as np
from matplotlib import pyplot as plt
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


def logistic(N: int,
             D1: DataParam,
             D2: DataParam,
             method: Literal["newton", "GD"] = "newton",
             max_iters=300,
             lr=1e-2):
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
    data[N:2*N, :2] = ds2
    label = np.zeros((2 * N, 1))
    label[N:] = 1

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.scatter(ds1[:, 0], ds1[:, 1], color='blue')
    plt.scatter(ds2[:, 0], ds2[:, 1], color='red')

    weight = np.ones((3,1))
    for it in range(max_iters):
        gradient = data.T @ (1 / (1 + np.exp(-data @ weight)) - label)
        weight -= lr * gradient
        if np.abs(gradient).sum() < 1e-8:
            break

    pred = np.squeeze(1 / (1 + np.exp(-data @ weight)))
    pred_0 = data[pred <= 0.5, :]
    pred_1 = data[pred > 0.5, :]
    plt.subplot(1, 3, 2)
    plt.scatter(pred_0[:, 0], pred_0[:, 1], color='blue')
    plt.scatter(pred_1[:, 0], pred_1[:, 1], color='red')

    plt.show()


if __name__ == "__main__":
    mx1 = 1
    my1 = 1
    vx1 = 2
    vy1 = 2
    mx2 = 10
    my2 = 10
    vx2 = 2
    vy2 = 2
    D1 = DataParam(mx1, my1, vx1, vy1)
    D2 = DataParam(mx2, my2, vx2, vy2)
    logistic(100, D1, D2)
