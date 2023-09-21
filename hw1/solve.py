import numpy as np
import pandas as pd


def matmul(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    """
    My own matmul implementation
    """
    assert len(a.shape) == len(b.shape)
    assert len(a.shape) == 2
    if a.shape[1] != b.shape[0]:
        raise ArithmeticError("dim 1 of a does not match dim 0 of b")
    m1, n = a.shape
    n, m2 = b.shape
    out = np.zeros((m1, m2), dtype=np.float32)
    for i in range(m1):
        for j in range(m2):
            for k in range(n):
                out[i, j] += a[i, k] * b[k, j]
    
    return out

def transpose(a:np.ndarray) -> np.ndarray:
    """
    Transpose the matrix
    """
    assert len(a.shape) == 2
    m, n = a.shape
    out = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            out[i, j] = a[j, i]
    return out

def LU_decompose(a:np.ndarray) -> np.ndarray:
    pass

def inverse(a:np.ndarray) -> np.ndarray:
    """
    find the inverse of matrix a using Gauss-Jordan
    """
    assert len(a.shape) == 2
    assert a.shape[0] == a.shape[1]

    n = a.shape[0]
    E = np.zeros((n, 2*n))
    E[:, n:] = np.identity(n)
    E[:, :n] = a
    for i in range(n):
        p = E[i,i]
        if p == 0:
            # exchange row
            for j in range(i+1, n):
                if E[j, i] != 0:
                    temp = np.copy(E[j])
                    E[j] = E[i]
                    E[i] = temp
                    break
            else:
                raise ArithmeticError("cannot find inverse")

        E[i] = E[i] / E[i,i]
        for j in range(i+1, n):
            E[j] = E[j] - E[j,i] * E[i]
    
    print(E)
    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            E[j] = E[j] - E[j,i] * E[i]

    return E[:, n:]




def closed_form(filename:str, n:int, lmda:float):
    df = pd.read_csv(filename)
    data = np.array(df)
    # print(data)
    N = data.shape[0]
    A = np.zeros((N, n))
    for i in range(n):
        A[:,i] = data[:, 0] ** np.array(i)
    b = data[:, 1]

def newton(filename:str, n:int, lmda:float):
    pass

def steepest_descent(filename:str, n:int, lmda:float):
    pass


if __name__ == "__main__":
    closed_form("testfile.txt", 3, 0)
    i = np.identity(4)

    test_mat = np.array([
        [0,1,4],
        [0,1,0],
        [-1,0,0],
        ])
    print(test_mat)
    test_mat_i = inverse(test_mat)
    print(test_mat_i)
    print(matmul(test_mat_i, test_mat))
