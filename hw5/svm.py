from typing import List
import numpy as np
import matplotlib.pyplot as plt
import os.path
from libsvm.svmutil import svm_train, svm_predict, svm_parameter, svm_problem
import time
from heatmap import annotate_heatmap, heatmap

def read_data(folder_path:str="./data/") -> List[np.ndarray]:
    files = ["X_train.csv", "Y_train.csv", "X_test.csv", "Y_test.csv"]
    data = []
    for fn in files:
        lines = None
        with open(os.path.join(folder_path, fn)) as f:
            lines = f.readlines()
        d = []
        for l in lines:
            nums = l.split(",")
            d.append(np.array([float(n) for n in nums]))
        d = np.array(d, dtype=np.float32).squeeze()
        data.append(d)
    data = [np.array(data[i]) for i in range(4)]
    return data

def task1(data:List[np.ndarray]):
    """
    Comparison of linear, polynomial and RBF kernels
    param data: the input data
    return: None
    """
    kernels = ['Linear', 'Polynomial', 'RBF']
    X_train, Y_train, X_test, Y_test = data

    # Get performance of each kernel
    for idx, name in enumerate(kernels):
        param = svm_parameter(f"-t {idx} -q")
        prob = svm_problem(Y_train, X_train)

        print(f'using {name} kernel')

        start = time.time()
        model = svm_train(prob, param)
        labels, acc, vals = svm_predict(Y_test, X_test, model, " -q")
        print(f"acc is {acc[0]}%")
        end = time.time()
        print(f'Elapsed time = {end - start:.2f}s\n')



def task2(data:List[np.ndarray]):
    """
    Grid search for best parameters of each kernel
    :param training_image: training images
    :param training_label: training labels
    :param testing_image: testing images
    :param testing_label: testing labels
    :return: None
    """

    kernels = ['Linear', 'Polynomial', 'RBF']
    X_train, Y_train, X_test, Y_test = data

    # Best parameters and max accuracies
    best_parameter = []
    max_accuracy = []

    # Find best parameters of each kernel
    for idx, name in enumerate(kernels):
        best_para = ''
        max_acc = 0.0
        if name == 'Linear':
            cost = [10 ** i for i in range(-1, 3)]
            for c in cost:
                parameters = f'-t {idx} -c {c}'
                acc = cross_val(X_train, Y_train, parameters)

                if acc > max_acc:
                    max_acc = acc
                    best_para = parameters
            best_parameter.append(best_para)
            max_accuracy.append(max_acc)
        elif name == 'Polynomial':
            # cost = [10 ** i for i in range(-1, 2)]
            # degree = [i for i in range(0, 3)]
            # gamma = [10 ** i for i in range(-1, 1)]
            # constant = [i for i in range(-1, 2)]
            cost = []
            degree = []
            gamma = []
            constant = []
            for c in cost:
                for d in degree:
                    for g in gamma:
                        for const in constant:
                            parameters = f'-t {idx} -c {c} -d {d} -g {g} -r {const}'
                            acc = cross_val(X_train, Y_train, parameters)

                            if acc > max_acc:
                                max_acc = acc
                                best_para = parameters
            best_parameter.append(best_para)
            max_accuracy.append(max_acc)
        elif name == 'RBF':
            cost = [2 ** i for i in range(-2, 6)]
            gamma = [2** i for i in range(-3, 7)]
            heat_data = []
            for c in cost:
                d_row = []
                for g in gamma:
                    parameters = f'-t {idx} -c {c} -g {g}'
                    acc = cross_val(X_train, Y_train, parameters)
                    d_row.append(acc)

                    if acc > max_acc:
                        max_acc = acc
                        best_para = parameters
                heat_data.append(d_row)
            best_parameter.append(best_para)
            max_accuracy.append(max_acc)
            fig, ax = plt.subplots()
            im, cbar = heatmap(heat_data, gamma, cost, ax=ax,
                               cmap="YlGn", cbarlabel="max cross-val accuracy")
            texts = annotate_heatmap(im, valfmt="{x:.2f}%")
            fig.tight_layout()
            plt.show()

    # Print results and prediction
    prob = svm_problem(Y_train, X_train)
    print()
    for idx, name in enumerate(kernels):
        print(f'using {name} kernel')
        print(f'max cross val acc: {max_accuracy[idx]}%')
        print(f'best parameters: {best_parameter[idx]}')
        print('validating...')

        model = svm_train(prob, svm_parameter(best_parameter[idx] + ' -q'))
        svm_predict(Y_test, X_test, model)
        print()


def cross_val(training_image: np.ndarray, training_label: np.ndarray, parameters: str,
                   is_kernel: bool = False) -> float:
    """
    Cross validation for the given kernel and parameters
    :param training_image: training images
    :param training_label: training labels
    :param parameters: given parameters
    :param is_kernel: whether training_image is actually a precomputed kernel
    :return: accuracy
    """
    param = svm_parameter(parameters + ' -v 4 -q') # v specified n-fold cross validation mode
    prob = svm_problem(training_label, training_image, isKernel=is_kernel)
    return svm_train(prob, param)


def task3(data:List[np.ndarray]) -> None:
    """
    Combination of linear and RBF kernels
    :param training_image: training images
    :param training_label: training labels
    :param testing_image: testing images
    :param testing_label: testing labels
    :return: None
    """
    # Parameters
    X_train, Y_train, X_test, Y_test = data
    cost = [np.power(10.0, i) for i in range(-2, 3)]
    gamma = [np.power(10.0, i) for i in range(-3, 3)]
    rows, _ = X_train.shape

    # Use grid search to find best parameters
    linear = linear_kernel(X_train, X_train)
    best_parameter = '-t 4'
    best_gamma = 1.0
    max_accuracy = 0.0
    for c in cost:
        for g in gamma:
            rbf = rbf_kernel(X_train, X_train, g)

            # The combination is linear + RBF, but np.arange is the required serial number from the library
            combination = np.hstack((np.arange(1, rows + 1).reshape(-1, 1), linear + rbf))

            parameters = f'-t 4 -c {c}'
            acc = cross_val(combination, Y_train, parameters, True)
            if acc > max_accuracy:
                max_accuracy = acc
                best_parameter = parameters
                best_gamma = g

    # Print best parameters and max accuracy
    print()
    print('using Linear + RBF')
    print(f'max validation acc: {max_accuracy}%')
    print(f'Best parameters: {best_parameter} -g {best_gamma}')

    # Train the model using best parameters
    rbf = rbf_kernel(X_train, X_train, best_gamma)
    combination = np.hstack((np.arange(1, rows + 1).reshape(-1, 1), linear + rbf))
    model = svm_train(svm_problem(Y_train, combination, isKernel=True), svm_parameter(best_parameter + ' -q'))

    # Make prediction using best parameters
    rows, _ = X_test.shape
    linear = linear_kernel(X_test, X_test)
    rbf = rbf_kernel(X_test, X_test, best_gamma)
    combination = np.hstack((np.arange(1, rows + 1).reshape(-1, 1), linear + rbf))
    svm_predict(Y_test, combination, model)


def linear_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Linear kernel, <x, y>
    :param x: point x
    :param y: point y
    :return: linear distance between them
    """
    assert x.shape == y.shape
    return x.dot(y.T)


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    """
    RBF kernel exp^(-gamma * ||x - y||^2)
    :param x: point x
    :param y: point y
    :param gamma: gamma coefficient
    :return: polynomial distance between them
    """
    assert x.shape == y.shape
    return np.exp(-gamma * np.sum((x-y) ** 2))


if __name__ == "__main__":
    a = read_data()
    # task1(a)
    task2(a)
    # task3(a)



