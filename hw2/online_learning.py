import numpy as np
from typing import List
import math


def C(N: int, m: int):
    return math.factorial(N) / (math.factorial(N - m) * math.factorial(m))


def online(data: List[str], a: float, b: float):
    for i, d in enumerate(data):
        nums = np.array([int(x) for x in list(d.strip("\n"))])
        be = beta(a, b)
        MLE_p = np.mean(nums)
        N = nums.shape[0]
        head = np.sum(nums).item()
        tail = nums.shape[0] - head
        likelihood = C(N, head) * MLE_p**head * (1 - MLE_p)**tail

        print("Case {}: {}".format(i + 1, d.strip("\n")))
        print("Likelihood: {}".format(likelihood))
        print("Beta prior: a={}, b={}".format(a, b))
        a += head
        b += tail
        print("Beta posterior: a={}, b={}".format(a, b))
        print()


if __name__ == "__main__":
    a: float = 10
    b: float = 1
    inputfile = "testfile.txt"

    data = []
    with open(inputfile, "r") as f:
        data = f.readlines()

    online(data, a, b)
