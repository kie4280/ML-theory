import numpy as np
from PIL import Image

def read_img(filename:str="image1.png") -> np.ndarray:
    img = np.asarray(Image.open(filename))
    return img


if __name__ == "__main__":
    im = read_img()
    print(im.shape)

