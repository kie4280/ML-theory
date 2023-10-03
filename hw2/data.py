import numpy as np
from typing import Tuple


def load_train(img_file:str = "train-images-idx3-ubyte", label_file: str = "train-labels-idx1-ubyte") -> Tuple[np.ndarray, np.ndarray]:

    with open(img_file, mode='rb') as f:
        f.seek(4)
        img_len = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big') # 28
        cols = int.from_bytes(f.read(4), byteorder='big') # 28
        img = np.array(bytearray(f.read(img_len * rows * cols)), dtype=np.uint8)
        img = img.reshape((img_len, 28*28))
    with open(label_file, mode='rb') as f:
        f.seek(4)
        la_len = int.from_bytes(f.read(4), byteorder='big')
        labels = np.array(bytearray(f.read(la_len)), dtype=np.uint8)

    return img, labels

def load_test(img_file:str = "t10k-images-idx3-ubyte", label_file: str = "t10k-labels-idx1-ubyte") -> Tuple[np.ndarray, np.ndarray]:

    with open(img_file, mode='rb') as f:
        f.seek(4)
        img_len = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big') # 28
        cols = int.from_bytes(f.read(4), byteorder='big') # 28
        img = np.array(bytearray(f.read(img_len * rows * cols)), dtype=np.uint8)
        img = img.reshape((img_len, 28*28))
    with open(label_file, mode='rb') as f:
        f.seek(4)
        la_len = int.from_bytes(f.read(4), byteorder='big')
        labels = np.array(bytearray(f.read(la_len)), dtype=np.uint8)

    return img, labels

