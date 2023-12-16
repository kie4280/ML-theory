from typing import Tuple
from matplotlib import os
import numpy as np
import matplotlib.pyplot as plt

COLOR = [[0, 102, 204], [51, 204, 204], [153, 102, 51], [153, 153, 153],
         [12, 23, 100], [145, 100, 0]]


class KMeans:

    def __init__(self) -> None:
        self._means = None
        pass

    def E_Step(self, means: np.ndarray):
        """
        :param means: the means of clusters
        Return: the new cluster assignment
        """
        cluster = np.zeros((self.U.shape[0], self.K), dtype=np.int32)

        for i in range(self.K):
            cluster[:, i] = np.sum(
                (self.U - np.expand_dims(means[i], axis=0))**2, axis=1)
        cluster = np.argmin(cluster, axis=1)

        return cluster

    def M_Step(self, cluster: np.ndarray) -> np.ndarray:
        """
        :param cluster: the cluster assignment
        Return: the new mean
        """
        m = np.zeros((self.K, self.U.shape[1]), dtype=np.float32)
        for i in range(self.K):
            mask = np.expand_dims(np.where(cluster == i, 1, 0), axis=1)
            n = np.sum(mask)
            m[i] = np.sum(self.U * mask, axis=0) / (n + 1e-8)

        return m

    def initial_mean(self, initType: str = "pick"):
        """
        :param X: (#datapoint,#eigenvectors) ndarray
        :param initType: 'pick', 'gaussian', 'k_means++'
        Return: initial mean, initial cluster assignment
        """

        m = np.zeros((self.K, self.U.shape[1]), dtype=np.float32)
        if initType == "kmeans++":
            pass
        elif initType == "pick":
            random_pick = np.random.choice(self.U.shape[0],
                                           size=self.K,
                                           replace=False)
            m = self.U[random_pick, :]
        elif initType == "gaussian":
            X_mean = np.mean(self.U, axis=0)
            X_std = np.std(self.U, axis=0)
            for i in range(self.K):
                m[:, i] = np.random.normal(X_mean[i], X_std[i], size=self.K)

        return m, np.ones((self.U.shape[0], ), dtype=np.int32) * -1

    def cluster(self,
                U: np.ndarray,
                img: np.ndarray,
                result_file_path: str = "./results",
                visualize: bool = False,
                num_cluster: int = -1,
                max_iters:int=1000) -> np.ndarray:
        """
        :param U: the laplacian matrix
        :param img: the input image (100, 100, 3)
        :param result_file_path: the location to store the visualization sequence
        Return: the clustering reults
        """

        os.makedirs(result_file_path, exist_ok=True)
        self.U = U
        self.K = U.shape[1] if num_cluster == -1 else num_cluster
        # Init means
        means, cluster = self.initial_mean(initType="pick")
        cluster_old = None
        delta = U.shape[0]
        _iter = 0

        # try:
        #     os.mkdir(result_file_path)
        # except:
        #     pass

        while delta > 0 and _iter <= max_iters:
            _iter += 1
            cluster_old = cluster.copy()

            # E Step: clustering
            cluster = self.E_Step(means)

            # M Step: update means
            means = self.M_Step(cluster)

            # Validate
            delta = np.sum((cluster_old != cluster))
            print(cluster)
            print(f"Iter:{_iter}, delta:{delta}")
            if visualize:
                self.visualize(img, cluster, _iter, result_file_path)

        if self.K < 4:
            pass
            # self.drawEigenspace(cluster, result_file_path)
        self._means = means

        return cluster

    def get_means(self):
        return self._means


    def visualize(self, img: np.ndarray, clusters: np.ndarray, _iter: int,
                  result_file_path: str) -> None:
        """
        :param img: the input image (100, 100, 3)
        :param clusters: the cluster assignment
        :param _iter: the number of current kmeans iteration
        :param result_file_path: the location to store the visualization sequence
        """
        im = np.array(COLOR, dtype=np.uint8)
        im = im[clusters.flatten(), :]
        im.resize((img.shape[0], img.shape[1], 3))
        print(im.shape)

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

    def drawEigenspace(self, cluster: np.ndarray, result_file_path: str):
        pt_x, pt_y, pt_z = [], [], []
        for k in range(self.K):
            pt_x.append([])
            pt_y.append([])
            pt_z.append([])
        fig = plt.figure()
        if self.K == 2:
            for n in range(img_size):
                pt_x[cluster[n]].append(self.U[n][0])
                pt_y[cluster[n]].append(self.U[n][1])
            for k in range(self.K):
                plt.scatter(pt_x[k], pt_y[k], c=COLOR[k], s=0.5)
        if self.K == 3:
            ax = fig.add_subplot(projection='3d')
            for n in range(img_size):
                pt_x[cluster[n]].append(self.U[n][0])
                pt_y[cluster[n]].append(self.U[n][1])
                pt_z[cluster[n]].append(self.U[n][2])
            for k in range(self.K):
                ax.scatter(pt_x[k], pt_y[k], pt_z[k], c=COLOR[k], s=0.5)
        plt.show()

        fig.savefig(f'{result_file_path}/eigen.jpg')


if __name__ == "__main__":
    pass
    k = KMeans()
    rng = np.random.default_rng()
    pts = np.zeros((900, 2), dtype=np.float32)

    for g in range(5):
        x = rng.normal(0, 1, size=(100))
        y = rng.normal(0, 1, size=(100))
        pts[g * 100:(g + 1) * 100] = np.stack([x, y], axis=1)
    plt.figure(dpi=300)
    plt.scatter(pts[:, 0], pts[:, 1])
    plt.show()
    cluster = k.cluster(pts,
                        np.zeros((30, 30, 3)),
                        visualize=False,
                        num_cluster=5)
    c = np.array(COLOR, dtype=np.float32) / 255
    plt.figure(dpi=300)
    plt.scatter(pts[:, 0], pts[:, 1], c=c[cluster])
    plt.show()
