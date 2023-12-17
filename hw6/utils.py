import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from typing import Tuple

COLOR = [[0, 102, 204], [51, 204, 204], [153, 102, 51], [153, 153, 153],
         [12, 23, 100], [145, 100, 0]]


def read_img(filename: str = "image1.png") -> np.ndarray:
    img = np.asarray(Image.open(filename).getdata())
    return img


def RBF_kernel(X: np.ndarray,
               gamma_c: float = 1 / (255 * 255),
               gamma_s: float = 1 / 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param x: the input image
    :param gamma_c: color similarity coef
    :param gamma_s: spatial similarity coef
    Return: kernel
    """
    dist_c = cdist(X, X, 'sqeuclidean')  #(10000,10000)

    seq = np.arange(0, 100)
    c_coord = seq
    for i in range(99):
        c_coord = np.hstack((c_coord, seq))
    c_coord = c_coord.reshape(-1, 1)
    r_coord = c_coord.reshape(100, 100).T.reshape(-1, 1)
    X_s = np.hstack((r_coord, c_coord))
    dist_s = cdist(X_s, X_s, 'sqeuclidean')

    RBF_s = np.exp(-gamma_s * dist_s)
    RBF_c = np.exp(-gamma_c * dist_c)  #(10000,10000)
    k = np.multiply(RBF_s, RBF_c)  #(10000,10000)

    return X_s, k


def visualize(im: np.ndarray,
              cluster: np.ndarray,
              iter: int,
              store: bool = False,
              show:bool = False,
              output_folder: str = "output"):

    im = np.array(COLOR)[cluster]
    im.resize((100, 100, 3))
    plt.figure(dpi=300)
    plt.imshow(im)
    if store:
        plt.savefig(f"{output_folder}/iter{iter}.png")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    im = read_img()
    print(im.shape)
