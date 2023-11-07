from typing import Tuple
import numpy as np
from data import load_test, load_train
from tqdm import trange


train_X, train_y = load_train()

IMG_SIZE = 28 * 28


def E_step(X: np.ndarray, lam: np.ndarray, prob: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    w = np.zeros((n, 10), dtype=np.float128)
    for c in range(10):
        x0 = X == 0
        x1 = X == 1
        p = lam[c]
        p = p * np.prod(x1 * prob[c] + x0 * 1, axis=1, dtype=np.float128)
        p = p * np.prod(x0 * (1 - prob[c]) + x1 * 1, axis=1, dtype=np.float128)
        # p *= np.prod(prob[c] ** X, axis=1, dtype=np.float128)
        # p *= np.prod((1-prob[c]) ** (1-X), axis=1, dtype=np.float128)
        # print(p.max())
        w[:, c] = p

    w = w / (w.sum(axis=1, keepdims=True) + 1e-620)
    return w


def M_step(X: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (lam, probs)
    """
    n = X.shape[0]
    lam = w.sum(axis=0) / n
    lam = lam / lam.sum()

    return lam, w.T @ X / np.expand_dims(w.sum(axis=0), axis=1)


def main(max_iters: int = 100):
    print(train_X.shape)
    grey = np.where(train_X < 128, 0, 1)
    # initial guess
    lam = np.random.uniform(0.4, 0.6, (10, ))
    prob = np.random.uniform(0.4, 0.6, (10, IMG_SIZE))
    prev_p = np.zeros((10, IMG_SIZE))

    for iter in trange(max_iters, ncols=80):
        w = E_step(grey, lam, prob)
        lam, prob = M_step(grey, w)
        print(w[0], lam, np.mean(np.abs(prev_p - prob)))

        if np.mean(np.abs(prev_p - prob)) < 1e-3:
            print(f"break after {iter + 1} iterations")
            break
        prev_p = prob


if __name__ == "__main__":
    main()
