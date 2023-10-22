import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def univar_gaussian(mean:float, var:float) -> float:
    U = np.random.uniform(0, 1)
    V = np.random.uniform(0, 1)
    z = np.sqrt(-2*np.log(U)) * np.cos(2*np.pi*V)
    z = (z + mean) * np.sqrt(var)
    return z.item()

def test_gaussian():
    plt.figure()
    x = np.zeros((10000))
    for i in range(10000):
        z = univar_gaussian(0, 1)
        x[i] = z

def poly_generator(weight:np.ndarray, n:int, var:float) -> Tuple[float, float]:
    x = np.random.uniform(-1, 1)
    weight = np.expand_dims(weight, axis=1)
    phi = np.array([x**i for i in range(n)])
    # print(weight.shape, phi.shape)
    y = np.matmul(weight.T, phi) + univar_gaussian(0, var)
    return x, y.item()
    
def test_poly():
    samples = 1000
    plt.figure()
    pts = np.zeros((samples, 2))
    for i in range(samples):
        x, y = poly_generator(np.array([1,1,1,]), 3, 0.001)
        pts[i] = np.array([x,y])

    plt.scatter(pts[:,0], pts[:,1])
    plt.show()

if __name__ == "__main__":
    # test_gaussian()
    test_poly()


    
