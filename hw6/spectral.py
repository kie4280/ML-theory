from matplotlib import os
import numpy as np
from utils import read_img, RBF_kernel, visualize, COLOR
from argparse import ArgumentParser
import time
import matplotlib.pyplot as plt
from typing import Tuple


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
        d = np.diag(d_diag)
        L = d @ L @ d
    return L, D


def kmeans(args, L:np.ndarray, img:np.ndarray) -> np.ndarray:
    """
    :param args: the program argument
    :param L: the input feature matrix
    :param img: the input image
    Return: cluster assignment
    """
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
        visualize(img, clusters, it, store=True, show=args.show, output_folder=args.output)
    return old_clusters

def initial_centers(L:np.ndarray, K:int, init_type:str="pick"):
    """
    :param L: the input feature matrix
    :param K: the number of cluster
    :param init_type: the initialization method
    Return: the initial mean
    """
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


def drawplot2D(args, data:np.ndarray, cluster:np.ndarray):
    K = args.clusters
    plt.figure(dpi=300)
    x = data[:, 0]
    y = data[:, 1]
    plt.xlabel("1st dim")
    plt.ylabel("2nd dim")
    plt.title("coordinates in the eigenspace of graph Laplacian")
    for i in range(K):
        plt.scatter(x[cluster==i], y[cluster==i], marker='.')
    # plt.show()
    plt.savefig(f"{args.img}_{args.method}_{K}.png")

def drawplot3D(args, data:np.ndarray, cluster:np.ndarray):
    K = args.clusters
    ax = plt.figure(dpi=300).add_subplot(projection="3d")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    ax.set_xlabel("1st dim")
    ax.set_ylabel("2nd dim")
    ax.set_zlabel("3rd dim")
    plt.title("coordinates in the eigenspace of graph Laplacian")
    for i in range(K):
        ax.scatter(x[cluster==i], y[cluster==i], z[cluster==i], '.')
    # plt.show()
    plt.savefig(f"{args.img}_{args.method}_{K}.png")


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
    parser.add_argument("--show", action="store_true")


    start_t = time.time()
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    img = read_img(args.img)
    # spatial, W = RBF_kernel(img.reshape(-1, 3), 0.001, 0.001)
    print("made kernel in {}".format(time.time() - start_t))
    if args.method == "normalized":

        if os.path.exists(f"eigen_{args.method}_{args.img}.npy"):
            eigen = np.load(f"eigen_{args.method}_{args.img}.npy")
            eigen /= np.sqrt(np.sum(eigen ** 2, axis=1)).reshape(-1, 1)
        else:
            L, _ = compute_LD(W, normalized=True)
            start_t = time.time()
            eigen = eigen_decompose(L)
            eigen /= np.sqrt(np.sum(eigen ** 2, axis=1)).reshape(-1, 1)
            np.save(f"eigen_{args.method}_{args.img}.npy", eigen)

    elif args.method == "ratio":

        if os.path.exists(f"eigen_{args.method}_{args.img}.npy"):
            eigen = np.load(f"eigen_{args.method}_{args.img}.npy")
        else:
            L, _ = compute_LD(W)
            start_t = time.time()
            eigen = eigen_decompose(L)
            np.save(f"eigen_{args.method}_{args.img}.npy", eigen)

    else:
        raise RuntimeError("unrecognized method")

    print("eigen decompose in {}".format(time.time() - start_t))
    eigen = eigen[:, 0:args.clusters]
    cluster = kmeans(args, eigen, img)
    if args.clusters == 2:
        drawplot2D(args, eigen, cluster)
    elif args.clusters == 3:
        drawplot3D(args, eigen, cluster)

    # im = np.array(COLOR)[cluster]
    # im.resize((100, 100, 3))
    # plt.figure(dpi=300)
    # plt.imshow(im)
    # plt.show()

if __name__ == "__main__":
    main()
