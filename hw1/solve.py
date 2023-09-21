import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    My own matmul implementation
    """

    if a.shape[-1] != b.shape[0]:
        raise ArithmeticError("dim -1 of a does not match dim 0 of b")
    n = a.shape[-1]
    if len(a.shape) == 1:
        a = np.expand_dims(a, axis=0)
    if len(b.shape) == 1:
        b = np.expand_dims(b, axis=1)
    m1, n = a.shape
    n, m2 = b.shape

    out = np.zeros((m1, m2), dtype=np.float32)
    for i in range(m1):
        for j in range(m2):
            for k in range(n):
                out[i, j] += a[i, k] * b[k, j]

    return np.squeeze(out)


def transpose(a: np.ndarray) -> np.ndarray:
    """
    Transpose the matrix
    """
    if len(a.shape) == 2:
        m, n = a.shape
        out = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                out[i, j] = a[j, i]
        return out
    elif len(a.shape) == 1:
        return np.expand_dims(a, axis=0)
    else:
        raise ArithmeticError("Unsupported transpose")


def inverse(a: np.ndarray) -> np.ndarray:
    """
    Find the inverse of matrix a using Gauss-Jordan
    """
    assert len(a.shape) == 2
    assert a.shape[0] == a.shape[1]

    n = a.shape[0]
    E = np.zeros((n, 2 * n))
    E[:, n:] = np.identity(n)
    E[:, :n] = a
    for i in range(n):
        p = E[i, i]
        if p == 0:
            for j in range(i + 1, n):
                if E[j, i] != 0:
                    # exchange row
                    temp = np.copy(E[j])
                    E[j] = E[i]
                    E[i] = temp
                    break
            else:
                raise ArithmeticError("cannot find inverse")

        E[i] = E[i] / E[i, i]
        for j in range(i + 1, n):
            E[j] = E[j] - E[j, i] * E[i]

    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            E[j] = E[j] - E[j, i] * E[i]

    return E[:, n:]


def poly_eval(x: np.ndarray, coefficients: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(n):
        out += coefficients[i] * (x**i)
    return out


def closed_form(filename: str, n: int, lmda: float):
    """
    Find the closed form solution for data using Least Squared Error and L2 norm.
    """
    df = pd.read_csv(filename)
    data = np.array(df)
    # print(data)
    N = data.shape[0]
    A = np.zeros((N, n))
    for i in range(n):
        A[:, i] = data[:, 0]**np.array(i)
    b = data[:, 1]
    AT = transpose(A)
    ata = matmul(AT, A)
    atb = matmul(AT, b)
    sol = matmul(inverse(ata + lmda * np.identity(n)), atb)
    line = [f"{float(sol[i])}x^{i}" for i in range(n - 1, 0, -1)]
    line.append(str(float(sol[0])))
    print("LSE:")
    print("bestfitting line: {}".format(" + ".join(line)))
    print("total error: {}".format(
        np.sum((poly_eval(data[:, 0], sol, n) - data[:, 1])**2)))
    print()
    plt.subplot(3, 1, 1)
    plt.scatter(data[:, 0], data[:, 1])
    x = np.linspace(-5, 5, 400)
    plt.plot(x, poly_eval(x, sol, n))
    plt.title("Closed form")


def newton(filename: str, n: int, lmda: float, iters:int=10):
    """
    Find the line that best fits the data using Newton's method
    """
    df = pd.read_csv(filename)
    data = np.array(df)
    # print(data)
    N = data.shape[0]
    A = np.zeros((N, n))
    for i in range(n):
        A[:, i] = data[:, 0]**np.array(i)
    b = data[:, 1]
    x = np.zeros((n))
    AT = transpose(A)

    for i in range(iters):
        gradient = 2 * (matmul(matmul(AT, A), x) - matmul(AT, b))
        hessian = 2 * matmul(AT, A)
        x = x - matmul(inverse(hessian), gradient)
    sol = x

    line = [f"{float(sol[i])}x^{i}" for i in range(n - 1, 0, -1)]
    line.append(str(float(sol[0])))
    print("Newton:")
    print("bestfitting line: {}".format(" + ".join(line)))
    print("total error: {}".format(
        np.sum((poly_eval(data[:, 0], sol, n) - data[:, 1])**2)))
    print()
    plt.subplot(3, 1, 2)
    plt.scatter(data[:, 0], data[:, 1])
    x = np.linspace(-5, 5, 400)
    plt.plot(x, poly_eval(x, sol, n))
    plt.title("Newton's method")

def steepest_descent(filename: str, n: int, lmda: float, iters:int=1000, lr:float = 1e-4):
    df = pd.read_csv(filename)
    data = np.array(df)
    # print(data)
    N = data.shape[0]
    A = np.zeros((N, n))
    for i in range(n):
        A[:, i] = data[:, 0]**np.array(i)
    b = data[:, 1]
    AT = transpose(A)
    x = np.zeros((n))
    for i in range(iters):
        x_sign = np.where(x>0, 1, -1)
        gradient = 2 * matmul((matmul(AT, A) + lmda * x_sign), x) - 2 * matmul(AT, b)
        # print(gradient)
        x = x - lr * gradient
    sol = x

    line = [f"{float(sol[i])}x^{i}" for i in range(n - 1, 0, -1)]
    line.append(str(float(sol[0])))
    print("Steepest descent:")
    print("bestfitting line: {}".format(" + ".join(line)))
    print("total error: {}".format(
        np.sum((poly_eval(data[:, 0], sol, n) - data[:, 1])**2)))
    print()
    plt.subplot(3, 1, 3)
    plt.scatter(data[:, 0], data[:, 1])
    x = np.linspace(-5, 5, 400)
    plt.plot(x, poly_eval(x, sol, n))
    plt.title("Steepest descent")


def testing():
    """
    This is for testing purposes only!
    Automated testing the funciton of inverse and matmul
    """
    I = np.identity(4)
    test_mat = np.identity(4)
    for i in range(100):
        swap_i = np.random.randint(0,4,(2,))
        amount = np.random.randn(4)
        test_mat[swap_i[0]] += test_mat[swap_i[1]] * amount
        test_mat_i = inverse(test_mat)
        res = matmul(test_mat_i, test_mat)
        s = np.sum(np.abs(res-I))
        if s > 1e-6:
            print(s)
            print(test_mat)
            print(res)

if __name__ == "__main__":
    plt.figure(dpi=200)

    # testing()
    filename = "testfile.txt"
    closed_form(filename, 3, 0)
    newton(filename, 3, 0)
    steepest_descent(filename, 3, 0)
    plt.show()
