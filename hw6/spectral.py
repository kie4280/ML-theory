from matplotlib import os
import numpy as np
from utils import read_img, RBF_kernel, visualize
from argparse import ArgumentParser
import time
import matplotlib.pyplot as plt
from typing import Tuple

COLOR = [[0, 102, 204], [51, 204, 204], [153, 102, 51], [153, 153, 153],
         [12, 23, 100], [145, 100, 0]]


def compute_LD(W: np.ndarray,
               normalized: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param w: the similarity matrix W
    Return: the degree of w
    """
    d_diag = np.sum(W, axis=1)
    D = np.diag(d_diag)
    L = D - W

    if normalized:
        d_diag = 1 / np.sqrt(d_diag)
        d = np.diagflat(d_diag)
        L = d @ L @ d
    return L, D


# def E_step(L: np.ndarray, means: np.ndarray, K: int = 2) -> np.ndarray:
#     dist = np.zeros((L.shape[0], K), dtype=np.float64)
#     for k in range(K):
#         m = means[k].reshape(1, -1)
#         dist[:, k] = np.linalg.norm(L - m, ord=2)
#     cluster = np.argmin(dist, axis=1)
#     return cluster


# def M_step(L: np.ndarray, cluster: np.ndarray, K: int = 2) -> np.ndarray:
#     new_center = np.zeros((K, L.shape[1]), dtype=np.float64)
#     for k in range(K):
#         mask = (cluster == k).reshape(-1, 1)
#         cluster_k = L * mask
#         print(cluster_k.shape, np.sum(mask))
#         new_center[k] = np.sum(cluster_k,
#                                axis=0, dtype=np.float64) / (np.sum(mask) + 1e-50)
#     return new_center


# def kmeans(args,
#            L: np.ndarray,
#            img: np.ndarray,
#            center: np.ndarray,
#            max_iters: int = 1000) -> np.ndarray:
#     K = args.clusters
#     cluster = E_step(L, center, K)
#     for it in range(max_iters):
#         print(np.sum(cluster))
#         visualize(img, cluster, it, store=True, output_folder=args.output)
#         print("old center", center)
#         center = M_step(L, cluster, K)
#         print("new center", center)
#         new_cluster = E_step(L, center, K)
#         diff_cluster = np.sum(new_cluster != cluster)
#         if diff_cluster == 0:
#             break
#         print("iter {} delta {}".format(it + 1, diff_cluster))
#         cluster = new_cluster
#     return cluster


# def initial_centers(X: np.ndarray,
#                     spatial_grid: np.ndarray,
#                     K: int = 2,
#                     init_type: str = "pick") -> np.ndarray:
#     """
#     :param X: (#datapoint,#features) ndarray
#     :param initType: 'pick', 'k_means++'
#     Return: initial cluster assignment
#     """

#     if init_type == "kmeans++":
#         center_idx = []
#         center_idx.append(np.random.choice(np.arange(10000), size=1)[0])
#         found = 1
#         while (found < K):
#             dist = np.zeros(10000)
#             for i in range(10000):
#                 min_dist = np.Inf
#                 for f in range(found):
#                     tmp = np.linalg.norm(spatial_grid[i, :] -
#                                          spatial_grid[center_idx[f], :])
#                     if tmp < min_dist:
#                         min_dist = tmp
#                 dist[i] = min_dist
#             dist = dist / np.sum(dist)
#             idx = np.random.choice(np.arange(10000), 1, p=dist)
#             center_idx.append(idx[0])
#             found += 1
#         center_idx = np.array(center_idx)
#         centers = X[center_idx, :]
#     elif init_type == "pick":
#         center_idx = np.random.choice(X.shape[0], size=K, replace=False)
#         centers = X[center_idx, :]
#     else:
#         center_idx = np.zeros(K)
#         centers = X[center_idx, :]

#     return centers

def kmeans(args, L:np.ndarray, img:np.ndarray) -> np.ndarray:
    init_type = args.init
    K = args.clusters

    mean = initial_centers(L, K, init_type)
    old_mean = np.zeros(mean.shape, dtype=L.dtype)
    it = 0
    old_clusters = np.zeros((L.shape[0]), dtype=np.int32)
    while np.linalg.norm(mean - old_mean) > 1e-10:
        # E-step: classify all samples
        clusters = np.zeros(L.shape[0], dtype=np.int32)
        for i in range(L.shape[0]):
            J = []
            for j in range(K):
                J.append(np.linalg.norm(L[i] - mean[j]))
            clusters[i] = np.argmin(J)

        # M-step: Update center mean
        old_mean = mean
        mean = np.zeros(mean.shape, dtype=L.dtype)
        counters = np.zeros(K)
        for i in range(L.shape[0]):
            mean[clusters[i]] += L[i]
            counters[clusters[i]] += 1
        for i in range(K):
            if counters[i] == 0:
                counters[i] = 1
            mean[i] /= counters[i]
        print("iter {} delta {}".format(it + 1, np.sum(old_clusters != clusters)))
        old_clusters = clusters
        it += 1
        visualize(img, clusters, it, store=True, output_folder=args.output)
    return old_clusters

def initial_centers(L:np.ndarray, K:int, init_type:str="pick"):
    mean = np.zeros((K, L.shape[1]), dtype=L.dtype) # mark
    if init_type == "pick": # normal k-means -> random center
        center = np.random.choice(10000, size=K, replace=False)
        mean = L[center,:]
    elif init_type == "kmeans++": # k-means++
        mean[0] = L[np.random.randint(L.shape[0], size=1), :]
        for cluste_id in range(1, K):
            temp_dist = np.zeros((len(L), cluste_id))
            for i in range(len(L)):
                for j in range(cluste_id):
                    temp_dist[i][j] = np.linalg.norm(L[i]-mean[j])
            dist = np.min(temp_dist, axis=1)
            sum = np.sum(dist) * np.random.rand()
            for i in range(len(L)):
                sum -= dist[i]
                if sum <= 0:
                    mean[cluste_id] = L[i]
                    break
    return mean


def eigen_decompose(L: np.ndarray) -> np.ndarray:
    """
    :param L: the matrix to decompose
    Return the eigenvectors without the one corresponding to eigenval=0
    """
    eigenvalue, eigenvector = np.linalg.eig(L)
    eigenindex = np.argsort(eigenvalue)  # sort according to the eigenvalues
    eigenvector = eigenvector[:, eigenindex]
    return eigenvector[:, 1:].real  # exclude first eigenvector


def main():
    parser = ArgumentParser()
    parser.add_argument("--clusters", "-c", default=2, type=int)
    parser.add_argument("--method",
                        "-m",
                        default="ratio",
                        choices=["normalized", "ratio"],
                        type=str)
    parser.add_argument("--init", default="pick", type=str)
    parser.add_argument("--output", "-o", default="./output", type=str)
    parser.add_argument("--img", default="image1.png", type=str)


    start_t = time.time()
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    img = read_img(args.img)
    spatial, W = RBF_kernel(img.reshape(-1, 3), 0.001, 0.001)
    print("made kernel in {}".format(time.time() - start_t))
    if args.method == "normalized":
        L, _ = compute_LD(W, normalized=True)
        np.save("lapla.npy", L)
        start_t = time.time()
        eigen = eigen_decompose(L)
        np.save("eigen.npy", eigen)
        pass
    elif args.method == "ratio":

        L = np.load("lapla.npy")
        eigen = np.load("eigen.npy")

        # L, _ = compute_LD(W)
        # np.save("lapla.npy", L)
        # start_t = time.time()
        # eigen = eigen_decompose(L)
        # np.save("eigen.npy", eigen)

        print("eigen decompose in {}".format(time.time() - start_t))
        eigen = eigen[:, 0:args.clusters]
        # init_c = initial_centers(eigen, spatial, args.clusters, args.init)
        # cluster = kmeans(args, eigen, img, init_c)
        cluster = kmeans(args, eigen, img)

        im = np.array(COLOR)[cluster]
        im.resize((100, 100, 3))
        plt.figure(dpi=300)
        # plt.imshow(im)
        # plt.show()


if __name__ == "__main__":
    main()
