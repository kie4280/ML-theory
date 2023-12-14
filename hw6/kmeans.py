from typing import Tuple
from matplotlib import os
import numpy as np
import matplotlib.pyplot as plt


def E_Step(U:np.ndarray, means:np.ndarray):
    """
    :param U:
    :param means: the means of clusters
    Return: the new cluster assignment
    """
    K = U.shape[1]
    cluster = np.zeros((U.shape[0], K), dtype=np.int32)

    for i in range(K):
        cluster[:, i] = np.sum((cluster - np.expand_dims(means[i], axis=0)) ** 2, axis=1)
    cluster = np.argmin(cluster, axis=1)

    return cluster
    
def M_Step(U:np.ndarray, cluster:np.ndarray) -> np.ndarray:
    """
    :param U: 
    :param cluster: the cluster assignment
    Return: the new mean
    """
    K = U.shape[1]
    m = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        mask = np.where(cluster == i, 1, 0)
        n = np.sum(mask)
        m[i] = U * mask / n

    return m 

def initial_mean(X:np.ndarray,initType:str="pick"):
    """
    :param X: (#datapoint,#eigenvectors) ndarray
    :param initType: 'pick', 'gaussian', 'k_means++'
    Return: initial mean, initial cluster assignment
    """

    K = X.shape[1]
    m = np.zeros((K, K), dtype=np.float32)
    if initType == "kmeans++":
        pass
    elif initType== "pick":
        random_pick = np.random.choice(X.shape[0], size=K, replace=False)
        m = X[random_pick, :]
    elif initType == "gaussian":
        X_mean=np.mean(X,axis=0)
        X_std=np.std(X,axis=0)
        for i in range(K):
            m[:,i] = np.random.normal(X_mean[i], X_std[i], size=K)

    return m, np.ones((X.shape[0], ), dtype=np.int32) * -1

def kmeans(U:np.ndarray, img:np.ndarray, k:int, result_file_path:str="./results") -> None:
    os.makedirs(result_file_path, exist_ok=True)
    # Init means
    means, cluster = initial_mean(U, initType="pick")
    cluster_old = None
    delta = U.shape[0]
    _iter = 0   
    
    # result_file_path = f'./sc result/img{IMAGE}_{K} class_{METHOD}_{CUT} cut'
    # try:
    #     os.mkdir(result_file_path)
    # except:
    #     pass
   
    while delta > 0:
        _iter += 1
        cluster_old = cluster.copy()
        
        # E Step: clustering
        cluster = E_Step(U, means)
        
        # M Step: update means
        means = M_Step(U, cluster)
        
        # Validate
        delta = np.sum((cluster_old != cluster))
        print(f"Iter:{_iter}, delta:{delta}")
        visualize(img, cluster, _iter, result_file_path)
        
    if K < 4:
        drawEigenspace(cluster, U, result_file_path)

def visiualize(img:np.ndarray, clusters:np.ndarray, _iter:int, result_file_path:str) -> None:
    im = np.zeros((img_length, img_length, 3), dtype=np.uint8)
    for n in range(img_size):
        im[n//img_length][n%img_length] = COLOR[K-2][cluster[n]]
    
    fig = plt.figure(figsize=(6, 4), dpi=300)
    # ax = fig.add_subplot(1, 2, 1)
    # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    plt.imshow(im)
    plt.title(f"Iteration: {_iter}")
    
    # ax = fig.add_subplot(1, 2, 2)
    # img = cv2.cvtColor(img.reshape(img_length, img_length, 3), cv2.COLOR_RGB2BGR)
    # ax.imshow(img)
    plt.savefig(f'{result_file_path}/{_iter}.jpg')
    plt.show()


if __name__ == "__main__":
    pass

