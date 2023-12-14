import numpy as np
from utils import read_img
from argparse import ArgumentParser


def make_kernel(data: np.ndarray,
                gamma_c: float = 1,
                gamma_s: float = 1) -> np.ndarray:
    """
    :param x: the input image
    :param gamma_c: color similarity coef
    :param gamma_s: spatial similarity coef
    Return: kernel
    """
    img = data.reshape(data.shape[0] * data.shape[1], 3)
    x = np.broadcast_to(np.expand_dims(img.astype(np.int32), axis=0),
                        (img.shape[0], img.shape[0], 3))
    idx = np.arange(img.shape[0])
    idx = np.expand_dims(np.stack([idx / img.shape[0], idx % img.shape[0]],
                                  axis=1),
                         axis=0)
    idx = np.broadcast_to(idx, (img.shape[0], img.shape[0], 2))
    color_sim = np.sum((x - np.moveaxis(x, 0, 1))**2, axis=2)
    spatial_sim = np.sum((idx - np.moveaxis(idx, 0, 1))**2, axis=2)
    K = np.exp(-gamma_s * spatial_sim - gamma_c * color_sim)
    # print(K.shape)

    return K


def compute_D(w: np.ndarray) -> np.ndarray:
    """
    :param w: the similarity matrix W
    Return: the degree of w
    """
    d = np.diagflat(np.sum(w, axis=1))
    return d

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
    
    args = parser.parse_args()
    img = read_img()
    W = make_kernel(img)
    if args.method == "normalized":
        pass
    elif args.method == "ratio":
        D = compute_D(W)
        L = D - W
        eigen = eigen_decompose(L)
        eigen = eigen[:, 0:args.c]
        print(eigen)

if __name__ == "__main__":
    main()
