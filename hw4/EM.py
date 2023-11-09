from typing import Tuple
import numpy as np
from data import load_test, load_train
from tqdm import trange

from utils import print_confusion


train_X, train_y = load_train()

IMG_SIZE = 28 * 28


def E_step(X: np.ndarray, lam: np.ndarray, prob: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    w = np.zeros((n, 10), dtype=np.float128)
    prob = np.where(prob<1e-5, 1e-5, prob)
    for r in range(n):
        for c in range(10):
            p = lam[c]
            p = p * np.prod((X[r]) * prob[c] + (1-X[r]) * (1-prob[c]), axis=0, dtype=np.float128)
            # p *= np.prod(prob[c] ** X, axis=1, dtype=np.float128)
            # p *= np.prod((1-prob[c]) ** (1-X), axis=1, dtype=np.float128)
            w[r, c] = p

    w = w / (w.sum(axis=1, keepdims=True) + 1e-620)
    return w


def M_step(X: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (lam, probs)
    """
    n = X.shape[0]
    lam = w.sum(axis=0) / n
    lam = lam / lam.sum()
    p = w.T @ X 

    return lam, p / np.expand_dims(w.sum(axis=0), axis=1)

def print_imagination(prob:np.ndarray):
    for c in range(10):
        print(f"class {c}")
        for i in range(IMG_SIZE):
            if i % 28 == 0:
                print()
                continue
            if prob[c,i] > 0.5:
                print("1",end='')
            else:
                print("0",end='')
             

        print("\n")

def assign_label(w:np.ndarray, true_label:np.ndarray):
    n = w.shape[0]
    w_max = np.argmax(w, axis=1)
    p_count = np.zeros((10, 10), dtype=np.int32)
    for i in range(n):
        p_count[w_max[i], true_label[i]] += 1
    return np.argmax(p_count, axis=1)

def main(max_iters: int = 5):
    grey = np.where(train_X < 128, 0, 1)
    # initial guess
    lam = 0.1 * np.ones((10,))
    prob = np.random.uniform(0.2, 0.8, (10, IMG_SIZE))
    prev_p = np.zeros((10, IMG_SIZE))

    w = np.zeros(1)
    for iter in trange(max_iters, ncols=80):
        w = E_step(grey, lam, prob)
        lam, prob = M_step(grey, w)
        # print(w[0], lam)

        if np.all(np.abs(prev_p - prob) < 1e-2) and iter > 5:
            print(f"break after {iter + 1} iterations")
            break
        # print("prob change {}".format(np.mean((np.abs(prev_p - prob)))))
        print_imagination(prob)
        prev_p = prob

    mapping = assign_label(w, train_y)
    print(mapping)
    # print_imagination(prob)
    print_confusion()

if __name__ == "__main__":
    main()
