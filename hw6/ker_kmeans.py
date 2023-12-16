from matplotlib import os
import numpy as np
from utils import read_img, RBF_kernel, visualize
from argparse import ArgumentParser
import time
import matplotlib.pyplot as plt


def construct_C(K: int, cluster: np.ndarray) -> np.ndarray:
    """
    compute the kernel distance
    :param K: number of clusters
    :param cluster: cluster assignment
    Return sum of kernels in this cluster
    """
    C = np.zeros(K, dtype=np.int32)
    for k in range(K):
        indicator = np.where(cluster == k, 1, 0)
        C[k] = np.sum(indicator)
    return C


def distance(C: np.ndarray, K: int, kernel: np.ndarray,
             cluster: np.ndarray) -> np.ndarray:
    """
    compute the kernel distance
    :param C: C_k in the slides
    :param K: number of clusters
    :param cluster: cluster assignment
    Return kernel distance (10000, K)
    """
    dist = np.ones((K, 10000), dtype=np.float32)  # k(x_j, x_j) = 1

    for k in range(K):
        alpha = np.where(cluster == k, 1, 0)
        a = alpha.reshape(-1, 1).T
        second_term = a @ kernel
        second_term = second_term * 2 / C[k]
        dist[k] -= second_term.flatten()

        a = alpha.reshape(-1, 1)
        third_term = np.sum(a.T @ kernel @ a)
        third_term /= (C[k]**2)
        dist[k] += third_term

    return dist.T


def clustering(kernel: np.ndarray, cluster: np.ndarray, K: int = 2):
    C = construct_C(K, cluster)
    dist = distance(C, K, kernel, cluster)
    new_cluster = np.argmin(dist, axis=1)

    return new_cluster


def initial_kmeans(X: np.ndarray,
                   spatial_grid:np.ndarray,
                   K: int = 2,
                   initType: str = "pick") -> np.ndarray:
    """
    :param X: (#datapoint,#features) ndarray
    :param initType: 'pick', 'k_means++'
    Return: initial cluster assignment
    """

    if initType == "kmeans++":
        center_idx = []
        center_idx.append(np.random.choice(np.arange(10000), size=1)[0])
        found = 1
        while (found<K):
            dist = np.zeros(10000)
            for i in range(10000):
                min_dist = np.Inf
                for f in range(found):
                    tmp = np.linalg.norm(spatial_grid[i,:]-spatial_grid[center_idx[f],:])
                    if tmp<min_dist:
                        min_dist = tmp
                dist[i] = min_dist
            dist = dist/np.sum(dist)
            idx = np.random.choice(np.arange(10000), 1, p=dist)
            center_idx.append(idx[0])
            found += 1
        center_idx = np.array(center_idx)
    elif initType == "pick":
        center_idx = np.random.choice(X.shape[0], size=K, replace=False)
    else:
        center_idx = np.zeros(K)

    cluster = np.ones((X.shape[0], ), dtype=np.int32) * 0
    for i in range(K):
        cluster[center_idx[i]] = i
    return cluster


def kmeans(args,
           kernel: np.ndarray,
           img: np.ndarray,
           cluster: np.ndarray,
           max_iters: int = 1000) -> np.ndarray:
    K = args.clusters
    for it in range(max_iters):
        new_cluster = clustering(kernel, cluster, K)
        diff_cluster = np.sum(new_cluster != cluster)
        if diff_cluster == 0:
            break
        print("iter {} delta {}".format(it + 1, diff_cluster))
        visualize(img, cluster, it, store=True, output_folder=args.output)
        cluster = new_cluster
    return cluster


def main():
    parser = ArgumentParser()
    parser.add_argument("--clusters", "-c", default=2, type=int)
    parser.add_argument("--init",
                        "-m",
                        default="pick",
                        type=str)
    parser.add_argument("--output", "-o", default="./output", type=str)
    parser.add_argument("--img", default="image1.png", type=str)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    start_t = time.time()
    img = read_img(args.img)
    spatial, W = RBF_kernel(img)

    # w_comp = np.load("../../MachineLearning/kernel.npy")
    # print(np.sum(np.abs(W-w_comp)))
    print("made kernel in {}".format(time.time() - start_t))

    init_cluster = initial_kmeans(W, spatial, args.clusters, initType=args.init)
    cluster = kmeans(args, W, img, init_cluster)


if __name__ == "__main__":
    main()
