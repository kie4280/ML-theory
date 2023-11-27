import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple


def read_data(data_loc: str = "./data/input.data") -> np.ndarray:
    data = None
    with open(data_loc) as f:
        data = f.readlines()
    if data == None:
        raise RuntimeError("data not found")
    train = np.zeros((len(data), 2))
    for i, l in enumerate(data):
        train[i] = np.array(l.split(" "))

    return train


def make_kernel(data: np.ndarray, length: float,
                scale_mixture: float):
    x = data
    n = x.shape[0]
    x = np.broadcast_to(x, (n, n))
    K = (1 + ((x - x.T)**2) /
               (2 * scale_mixture * length * length))**(-scale_mixture)

    return K


def GP(params: Tuple,
       data: np.ndarray,
       beta: float = 5,
       points: int = 200,
       visualization: bool = False):
    length, scale_mixture = params
    # print(var, length, scale_mixture, data, beta, points, visualization)
    x_plot = np.linspace(-60, 60, points)
    N = data.shape[0]
    new_d = np.concatenate([data[:, 0], x_plot], axis=0)
    C = make_kernel(new_d, length,
                    scale_mixture) + 1 / beta * np.identity(N + points)
    y = data[:, 1]
    means = []
    vars = []
    C_inv = np.linalg.inv(C[:N, :N])
    for i in range(points):
        k = C[N + i, :N]
        m = k.T @ C_inv @ y
        v = C[N + i, N + i] - k.T @ C_inv @ k

        means.append(m)
        vars.append(v)
    means = np.array(means)
    std = np.sqrt(np.array(vars))
    if visualization:
        plt.figure(dpi=300)
        plt.fill_between(x_plot, means - 2 * std, means + 2 * std)
        plt.plot(x_plot, means, c="green")
        plt.scatter(data[:, 0], data[:, 1], marker='o', c="red")
        plt.show()

    C_N = C[:N, :N]
    return -1 / 2 * np.log(np.linalg.det(
        C_N)) - 1 / 2 * y.T @ C_inv @ y - N / 2 * np.log(2 * np.pi)


def optimize_param(data: np.ndarray, beta: float = 5):
    mini = minimize(GP, x0=np.random.uniform(5,10,(2, )), args=(data, beta, 200, False))

    length, scale_mixture = mini.x
    print(f"optimal is length {length}, scale_mixture {scale_mixture}")
    return mini.x


if __name__ == "__main__":
    train = read_data()
    i = GP((1,1),train, visualization=True)
    # print(i)
    i = optimize_param(train)
    GP(i, train, visualization=True)
