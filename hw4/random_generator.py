import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def univar_gaussian(mean:float, var:float) -> float:
    U = np.random.uniform(0, 1)
    V = np.random.uniform(0, 1)
    z = np.sqrt(-2*np.log(U)) * np.cos(2*np.pi*V)
    z = z * np.sqrt(var) + mean
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

def sequential_estimator(m:float, s:float, iters:int=100000):
    e_mean = 0
    e_var = 0
    n = 0
    print("Data point source function: N({}, {})".format(m, s))
    for i in range(iters):
        z = univar_gaussian(m, s)
        if n == 0:
            e_var = 0
        else:
            e_var = (e_var + e_mean**2) * ((n-1)/n) + (1/n) * z**2
        e_mean = e_mean * (n/(n+1)) + (1/(n+1)) * z
        e_var  = e_var - e_mean**2
        print("Add data point {}".format(z))
        print("Mean: {}, variance: {}".format(e_mean, e_var))
        n += 1
       

def draw_graph(points:np.ndarray, weight:np.ndarray, a:float, mean_list:List[np.ndarray], var_list:List[np.ndarray]):
    plt.figure()
    plt.subplot(2,2,1)
    x_GT = np.linspace(-2, 2, 100)
    phi = np.array([x_GT ** i for i in range(weight.shape[0])]).T
    y_GT = phi @ weight
    plt.plot(x_GT, y_GT, color="black")
    plt.plot(x_GT, y_GT - a, color="red")
    plt.plot(x_GT, y_GT + a, color="red")
    plt.title("GT")

    plt.subplot(2,2,2)
    plt.scatter(points[:,0], points[:,1])
    y = (phi @ mean_list[2]).squeeze()
    s = np.diag(phi @ var_list[2] @ phi.T)
    plt.plot(x_GT, y, color="black")
    plt.plot(x_GT, y - a - s, color="red")
    plt.plot(x_GT, y + a + s, color="red")
    plt.title("Prediction result")

    plt.subplot(2,2,3)
    plt.scatter(points[:10,0], points[:10,1])
    y = (phi @ mean_list[0]).squeeze()
    s = np.diag(phi @ var_list[0] @ phi.T)
    plt.plot(x_GT, y, color="black")
    plt.plot(x_GT, y - a - s, color="red")
    plt.plot(x_GT, y + a + s, color="red")
    plt.title("After 10 samples")

    plt.subplot(2,2,4)
    plt.scatter(points[:50,0], points[:50,1])
    y = (phi @ mean_list[1]).squeeze()
    s = np.diag(phi @ var_list[1] @ phi.T)
    plt.plot(x_GT, y, color="black")
    plt.plot(x_GT, y - a - s, color="red")
    plt.plot(x_GT, y + a + s, color="red")
    plt.title("After 50 samples")

    plt.show()

def baysian_LR(b:float, n:int, a:float, weight:np.ndarray, iters:int=300):
    points = []
    prior_mean = np.zeros((n,1))
    prior_var = (1/b) * np.identity(n)
    # tests = [[-0.64152, 0.19039],[0.07122, 1.63175]] # sanity check
    mean_list = []
    var_list = []
    for i in range(iters):
        x,y = poly_generator(weight, n, a)
        points.append((x,y))
        # x, y = tests[i]# sanity check
        phi = np.expand_dims(np.array([x**i for i in range(n)]), axis=0)
        var_inv = np.linalg.inv(prior_var)
        posterior_var = np.linalg.inv(var_inv + 1/a * phi.T @ phi)
        posterior_mean = posterior_var @ (1/a * phi.T * y + var_inv @ prior_mean)

        print("Add data point: ({}, {})".format(x, y))
        print("Posterior mean:")
        print(posterior_mean)
        print()
        print("Posterior variance:")
        print(posterior_var)
        print()
        marginalize_mean = phi @ prior_mean
        marginalize_var = 1/a + phi @ prior_var @ phi.T
        print("Predictive distribution ~ N({}, {})".format(marginalize_mean.item(), marginalize_var.item()))
        print()
        if i == 9 or i == 49:
            mean_list.append(prior_mean)
            var_list.append(prior_var)

        prior_mean = posterior_mean
        prior_var = posterior_var

    mean_list.append(prior_mean)
    var_list.append(prior_var)
    draw_graph(np.array(points), weight, a, mean_list, var_list)
        


if __name__ == "__main__":
    pass
    # test_gaussian()
    # test_poly()
    # sequential_estimator(0,5)
    baysian_LR(1, 3, 3, np.array([1,2,3]))


    
