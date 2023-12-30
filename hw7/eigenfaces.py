import os
from typing import Tuple, Literal, List
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange
from argparse import ArgumentParser
from glob import glob
from PIL import Image
from pathlib import Path
from scipy.spatial.distance import cdist

subject_num = 15
image_num = 11
train_num = 9
test_num = 2

# height, width = 231, 195
height, width = 50, 50
expression = [
    'centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal',
    'rightlight', 'sad', 'sleepy', 'surprised', 'wink'
]


def load_faces(
        mode: Literal["Training", "Testing"],
        root="./Yale_Face_Database") -> Tuple[np.ndarray, List[str], np.ndarray]:
  """
  :param mode: Training or Testing mode
  :param root: the root folder of the data
  Return list of imgs, list of filenames
  """
  files = sorted(glob(os.path.join(root, mode, "**")))
  imgs = []
  labels = []
  file_names = []
  for file in files:
    im = Image.open(file)
    im = im.resize((width, height), Image.BICUBIC)
    im = np.array(im)
    # im = np.moveaxis(im, 0, 1)
    imgs.append(im)
    filename = Path(file).stem
    file_names.append(filename)
    features = filename.split(".")
    l = int(features[0].replace("subject", ""))
    labels.append(l)

  imgs = np.array(imgs, dtype=np.int32)
  labels = np.array(labels, dtype=np.int32)

  return imgs, file_names, labels


def PCA(data: np.ndarray, k=25) -> Tuple[np.ndarray, np.ndarray]:
  mean = np.mean(data, axis=0)
  cov = (data - mean) @ (data - mean).T
  eigenvalue, eigenvector = np.linalg.eig(cov)
  eigenvector = data.T @ eigenvector

  # Normalize w
  for i in range(len(eigenvector[0])):
    eigenvector[:,
                i] = eigenvector[:, i] / np.linalg.norm(eigenvector[:, i])

  # Seclect first k largest eigenvalues
  eigenindex = np.argsort(-eigenvalue)
  eigenvector = eigenvector[:, eigenindex]

  W = eigenvector[:, :k].real

  return W, mean


def imageCompression(data, S) -> np.ndarray:
  d = np.zeros((len(data), height // S, width // S))
  for n in range(len(data)):
    d[n] = np.full((height // S, width // S), np.mean(data[n]))
    img = data[n].reshape(height, width)
    for i in range(0, height - S + 1, S):
      for j in range(0, width - S + 1, S):
        tmp = 0
        # Summation SxS area in original image
        for r in range(S):
          for c in range(S):
            tmp += img[i + r][j + c]
        # New value is the avg. value of SxS area in original image
        d[n][i // S][j // S] = tmp // (S**2)
  return d.reshape(len(data), -1)


# def LDA(data, label, dim):
#   N = data.shape[1]  # N pixels
#   mean = np.mean(data, axis=0)  # data mean (H*W, )
#   Sw = np.zeros((N, N))
#   Sb = np.zeros((N, N))
#   for i in range(15):  # 15 subjects
#     print("sdff", np.where(label == i+1)[0])
#     data_i = data[np.where(label == i+1)[0], :]
#     cmean_i = np.mean(data_i, axis=0)  # class mean
#     Sw += (data_i - cmean_i).T @ (data_i - cmean_i)
#     Sb += data_i.shape[0] * ((cmean_i - mean).T @ (cmean_i - mean))

#   eigenvalue, eigenvector = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
#   for i in range(eigenvector.shape[1]):
#     eigenvector[:, i] = eigenvector[:, i] / np.linalg.norm(eigenvector[:, i])
#   idx = np.argsort(eigenvalue)[::-1]
#   print(eigenvalue[idx][:20])
#   # W = eigenvector[:, idx][:, :dims].real
#   W = eigenvector[:, idx][:, :dim].real
#   return W, mean

def LDA(data:np.ndarray, label:np.ndarray, dims:int) -> Tuple[np.ndarray, np.ndarray]:
  """
  :param data: the input data
  :param label: the label of the face
  :param dims of the eigenmatrix to return
  Return largest eigenvector, mean of the data
  """
  data = data.astype(np.float64)
  (n, d) = data.shape
  label = np.asarray(label)
  c = np.unique(label)
  mu = np.mean(data, axis=0).reshape(1, -1)
  S_w = np.zeros((d, d), dtype=np.float64)
  S_b = np.zeros((d, d), dtype=np.float64)
  for i in c:
      X_i = data[label == i, :]
      mu_i = np.mean(X_i, axis=0).reshape(1, -1)
      S_w += (X_i - mu_i).T @ (X_i - mu_i)
      S_b += X_i.shape[0] * ((mu_i - mu).T @ (mu_i - mu))
  eigenvalue, eigenvector = np.linalg.eig(np.linalg.pinv(S_w) @ S_b)
  # for i in range(eigenvector.shape[1]):
  #     eigenvector[:, i] = eigenvector[:, i] / np.linalg.norm(eigenvector[:, i])
  idx = np.argsort(eigenvalue)[::-1]
  print(eigenvalue[idx])
  # W = eigenvector[:, idx][:, :dims].real
  W = eigenvector[:, idx][:, :dims].real
  return W, mu


def computeKernel(datai, dataj, _type, gamma=1e-7, c=1, d=2) -> np.ndarray:
  if _type == 'linear':
    return datai @ dataj.T
  elif _type == 'poly':
    return (gamma * (datai @ dataj.T) + c)**d
  elif _type == 'rbf':
    dist = cdist(datai, dataj, 'sqeuclidean')
    K = np.exp(-gamma * dist)
    return K
  else:
    raise RuntimeError("kernel not found")


def centered(K: np.ndarray) -> np.ndarray:
  """
  This formula is in the slides
  :param K: the kernel to be centered
  Return the centered kernel
  """
  n = K.shape[0]
  _1N = np.full((n, n), 1 / n)
  KC = K - _1N @ K - K @ _1N + _1N @ K @ _1N
  return KC


def kernelPCA(data: np.ndarray,
              kernel_type: str,
              dims=25) -> Tuple[np.ndarray, np.ndarray]:

  K = computeKernel(data, data, kernel_type)
  K = centered(K)  # center K using formula found in slides

  eigenvalue, eigenvector = np.linalg.eig(K)
  for i in range(len(eigenvector[0])):
    eigenvector[:,
                i] = eigenvector[:, i] / np.linalg.norm(eigenvector[:, i])
  eigenindex = np.argsort(-eigenvalue)
  eigenvector = eigenvector[:, eigenindex]
  W = eigenvector[:, :dims].real

  return W, K


def kernelLDA(data: np.ndarray, kernel_type: str, dims=25):
  Z = np.full((len(data), len(data)), 1 / train_num)
  K = computeKernel(data, data, kernel_type)

  Sw = K @ K
  Sb = K @ Z @ K

  # Pseudo inv.
  eigenvalue, eigenvector = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
  for i in range(len(eigenvector[0])):
    eigenvector[:,
                i] = eigenvector[:, i] / np.linalg.norm(eigenvector[:, i])
  eigenindex = np.argsort(-eigenvalue)
  eigenvector = eigenvector[:, eigenindex]
  W = eigenvector[:, :dims].real

  return W, K


def eigenFace(W: np.ndarray, file_path: str, k=25, S=1, show=False):
  fig = plt.figure()
  for i in range(k):
    img = W[:, i].reshape(height // S, width // S)
    print(img.max(), img.min())
    plt.imshow(img, cmap='gray')
    plt.savefig(f'{file_path}/eigenface_{i:02d}.jpg')
  plt.close()

  fig = plt.figure(figsize=(12, 9), dpi=300)
  for i in range(k):
    img = W[:, i].reshape(height // S, width // S)
    row = int(np.sqrt(k))
    ax = fig.add_subplot(row, row, i + 1)
    ax.imshow(img, cmap='gray')
  plt.savefig(f'{file_path}/eigenfaces_{k}.jpg')
  if show:
    plt.show()
  plt.close()


def reconstructFace(W, mean, data, file_path, S=1, show=False):
  if mean is None:
    mean = np.zeros(W.shape[0])

  sel = np.random.choice(subject_num * train_num, 10, replace=False)
  img = []
  for index in sel:
    x = data[index].reshape(1, -1)
    reconstruct = (x - mean) @ W @ W.T + mean
    img.append(reconstruct.reshape(height // S, width // S))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x.reshape(height // S, width // S),
                 cmap='gray')  # Original face
    ax[1].imshow(reconstruct.reshape(height // S, width // S),
                 cmap='gray')  # Reconstruct face
    fig.savefig(f'{file_path}/reconfaces_{len(img)}.jpg')
  plt.close()

  # Put all reconstruct faces together
  fig = plt.figure(figsize=(10, 4), dpi=300)
  for i in range(len(img)):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(img[i], cmap='gray')
  plt.savefig(f'{file_path}/reconfaces.jpg')
  if show:
    plt.show()
  plt.close()


def distance(test, train_data):
  """
  :param test: the reference data
  :param train_data
  """
  dist = np.zeros(len(train_data), dtype=np.float32)
  for j in range(len(train_data)):
    dist[j] = np.sum((test - train_data[j])**2)  # Euclidean distance
  return dist


def faceRecongnition(W, mean, train_data, test_data, K):
  """
  Do the KNN to get the classification results
  :param
  """
  if mean is None:
    mean = np.zeros(W.shape[0])

  # KNN
  err = 0
  low_train = (train_data - mean) @ W
  low_test = (test_data - mean) @ W
  for i in range(low_test.shape[0]):
    vote = np.zeros(subject_num, dtype=np.int32)
    dist = distance(low_test[i],
                    low_train)  # Compute distance to all train_data
    nearest = np.argsort(dist)[:K]
    for n in nearest:
      vote[n // train_num] += 1
    predict = np.argmax(vote) + 1
    if predict != i // 2 + 1:
      err += 1
  print(
      f"K={K}: Accuracy:{1 - err/len(low_test):.4f} ({len(low_test) - err}/{len(low_test)})"
  )
  return 1 - err / len(low_test)


def kernelFaceRecongnition(W: np.ndarray, train_data: np.ndarray,
                           test_data: np.ndarray, kernel_type: str, kernel: np.ndarray, K):
  low_train = kernel @ W

  K_test = computeKernel(test_data, train_data, kernel_type)
  low_test = K_test @ W

  # KNN
  err = 0
  for i in range(low_test.shape[0]):
    vote = np.zeros(subject_num, dtype=int)
    dist = distance(low_test[i], low_train)
    nearest = np.argsort(dist)[:K]
    for n in nearest:
      vote[n // train_num] += 1
    predict = np.argmax(vote) + 1
    # print(i // 2 + 1, predict, vote)
    if predict != i // 2 + 1:
      err += 1
  acc = 1 - err / low_test.shape[0]
  print(
      f"K={K}: Accuracy:{acc} ({low_test.shape[0] - err}/{low_test.shape[0]})")
  return acc


if __name__ == "__main__":
  acc = 0

  train_data, train_filenames, train_labels = load_faces("Training")
  test_data, test_filenames, test_labels = load_faces("Testing")
  train_data = np.reshape(train_data,
                          (train_data.shape[0], -1)).astype(np.float32)
  test_data = np.reshape(test_data,
                         (test_data.shape[0], -1)).astype(np.float32)

  parser = ArgumentParser()
  parser.add_argument("--mode", type=str, default="PCA")
  parser.add_argument("--dims", type=int, default=25)
  parser.add_argument("--show", action="store_true")
  parser.add_argument("--kernel",
                      type=str,
                      default="rbf",
                      choices=["rbf", "linear", "poly"])
  parser.add_argument("--output", type=str, default="output")
  args = parser.parse_args()

  # PCA
  if args.mode == "PCA":
    PCA_file = os.path.join(args.output, "PCA_LDA", "PCA")
    eigenface_path = os.path.join(PCA_file, "eigenfaces")
    reconstruct_path = os.path.join(PCA_file, "reconstruct")
    os.makedirs(eigenface_path, exist_ok=True)
    os.makedirs(reconstruct_path, exist_ok=True)
    W_PCA, mean_PCA = PCA(train_data, k=args.dims)
    eigenFace(W_PCA, eigenface_path, k=args.dims, show=args.show)
    reconstructFace(
        W_PCA,
        mean_PCA,
        train_data,
        reconstruct_path,
        show=args.show)
    for i in range(1, 20, 2):
      acc += faceRecongnition(W_PCA, mean_PCA, train_data, test_data, i)

  # LDA
  elif args.mode == "LDA":
    LDA_file = os.path.join(args.output, "PCA_LDA", "LDA")
    fisher_path = os.path.join(LDA_file, "fisherfaces")
    reconstruct_path = os.path.join(LDA_file, "reconstruct")
    os.makedirs(fisher_path, exist_ok=True)
    os.makedirs(reconstruct_path, exist_ok=True)
    # data = imageCompression(train_data, scalar)
    # compress_test = imageCompression(test_data, scalar)

    W_LDA, mean_LDA = LDA(train_data, train_labels, args.dims)
    eigenFace(W_LDA, fisher_path, k=args.dims, show=args.show)
    reconstructFace(
        W_LDA,
        mean_LDA,
        train_data,
        reconstruct_path,
        show=args.show)
    for i in range(1, 20, 2):
      acc += faceRecongnition(W_LDA, None, train_data, test_data, i)

  # Kernel PCA
  elif args.mode == "kernelPCA":
    avgFace = np.mean(train_data, axis=0)
    centered_train = train_data - avgFace
    centered_test = test_data - avgFace

    W_kPCA, kernel = kernelPCA(centered_train, args.kernel, dims=args.dims)
    for i in range(1, 20, 2):
      acc += kernelFaceRecongnition(W_kPCA, centered_train,
                                    centered_test, args.kernel, kernel,
                                    i)

  # Kernel LDA
  elif args.mode == "kernelLDA":
    avgFace = np.mean(train_data, axis=0)
    centered_train = train_data - avgFace
    centered_test = test_data - avgFace

    W_kLDA, kernel = kernelLDA(centered_train, args.kernel)
    for i in range(1, 20, 2):
      acc += kernelFaceRecongnition(W_kLDA, centered_train,
                                    centered_test, args.kernel, kernel,
                                    i)

  print(f"Average accuracy:{acc / 10: .4f}")
