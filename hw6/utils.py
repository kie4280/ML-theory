import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

COLOR = [[0, 102, 204], [51, 204, 204], [153, 102, 51], [153, 153, 153],
         [12, 23, 100], [145, 100, 0]]


def read_img(filename: str = "image1.png") -> np.ndarray:
    img = np.asarray(Image.open(filename).getdata())
    return img


def make_kernel(data: np.ndarray,
                gamma_c: float = 1 / (255 * 255),
                gamma_s: float = 1 / 10000) -> np.ndarray:
    """
    :param x: the input image
    :param gamma_c: color similarity coef
    :param gamma_s: spatial similarity coef
    Return: kernel
    """
    img = data.reshape(data.shape[0] * data.shape[1], 3).astype(np.int32)
    x = np.broadcast_to(np.expand_dims(img, axis=0),
                        (img.shape[0], img.shape[0], 3))
    idx = np.arange(img.shape[0], dtype=np.int32)
    idx = np.expand_dims(np.stack([idx / img.shape[0], idx % img.shape[0]],
                                  axis=1),
                         axis=0)
    idx = np.broadcast_to(idx, (img.shape[0], img.shape[0], 2))
    color_sim = np.sum((x - np.moveaxis(x, 0, 1))**2, axis=2)
    spatial_sim = np.sum((idx - np.moveaxis(idx, 0, 1))**2, axis=2)
    K = np.exp(-gamma_s * spatial_sim - gamma_c * color_sim)
    # print(K.shape)

    return K


def RBF_kernel(X: np.ndarray,
               gamma_c: float = 1 / (255 * 255),
               gamma_s: float = 1 / 10000) -> np.ndarray:
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

    return k


def visualize(im: np.ndarray,
              cluster: np.ndarray,
              iter: int,
              store: bool = False,
              output_folder: str = "output"):

    im = np.array(COLOR)[cluster]
    im.resize((100, 100, 3))
    plt.figure(dpi=300)
    plt.imshow(im)
    if store:
        plt.savefig(f"{output_folder}/iter{iter}.png")
    plt.show()


if __name__ == "__main__":
    im = read_img()
    print(im.shape)
