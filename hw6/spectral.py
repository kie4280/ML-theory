import numpy as np
from utils import read_img, make_kernel, RBF_kernel
from argparse import ArgumentParser
from kmeans import KMeans
import time
import matplotlib.pyplot as plt
from typing import Tuple

COLOR = [[0, 102, 204], [51, 204, 204], [153, 102, 51], [153, 153, 153],
         [12, 23, 100], [145, 100, 0]]

def compute_LD(W: np.ndarray, normalized:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param w: the similarity matrix W
    Return: the degree of w
    """
    d_diag = np.sum(W, axis=1)
    D = np.diagflat(d_diag)
    L = D - W

    if normalized:
        d_diag = 1 / np.sqrt(d_diag)
        d = np.diagflat(d_diag) 
        L = d @ L @ d
    return L, D

def eigen_decompose(L:np.ndarray) -> np.ndarray:
    """
    :param L: the matrix to decompose
    Return the eigenvectors without the one corresponding to eigenval=0
    """
    eigenvalue, eigenvector = np.linalg.eig(L)
    eigenindex = np.argsort(eigenvalue) # sort according to the eigenvalues 
    eigenvector = eigenvector[:, eigenindex]
    return eigenvector[:, 1:].real # exclude first eigenvector

def main():
    parser = ArgumentParser()
    parser.add_argument("--clusters", "-c", default=2, type=int)
    parser.add_argument("--method",
                        "-m",
                        default="ratio",
                        choices=["normalized", "ratio"],
                        type=str)
    
    start_t = time.time()
    args = parser.parse_args()
    img = read_img()
    W = RBF_kernel(img.reshape(-1, 3))
    print("made kernel in {}".format(time.time() - start_t))
    if args.method == "normalized":
        L, D = compute_LD(W, normalized=True)
        pass
    elif args.method == "ratio":
        L, D = compute_LD(W)
        
        np.save("lapla.npy", L)
        # L = np.load("lapla.npy")
        start_t = time.time()
        eigen = eigen_decompose(L)
        np.save("eigen.npy", eigen)
        # eigen = np.load("eigen.npy")
        print("eigen decompose in {}".format(time.time() - start_t))
        eigen = eigen[:, 0:args.clusters]
        print(eigen[:,0].min(), eigen[:,0].max())
        k = KMeans()
        cluster = k.cluster(eigen, img, max_iters=100)
        print(k.get_means())
        im = np.array(COLOR)[cluster]
        im.resize(img.shape)
        plt.figure(dpi=300)
        plt.imshow(im)
        plt.show()
        print(np.sum(cluster))


if __name__ == "__main__":
    main()
