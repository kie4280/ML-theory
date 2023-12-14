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
