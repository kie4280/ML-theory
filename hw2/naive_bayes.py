from enum import Enum
from typing import Tuple
import data
import numpy as np

test_X, test_y = data.load_test()
train_X, train_y = data.load_train()


class Mode(Enum):
    DISCRETE = 0
    CONTINUOUS = 1

# Taken from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()

def discrete(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # prior
    prior_train = np.mean(np.array(
        [np.where(train_y == x, 1, 0) for x in range(10)]),
                          axis=1)
    # print(prior_train)
    # print(np.sum(prior_train))

    # likelihood
    pred_y = np.zeros((test_X.shape[0], ))

    likelihood = np.zeros((10, 32, 28 * 28))  # log likelihood

    for label in range(10):
        label_mask = np.where(train_y == label, True, False)
        match_la = train_X[label_mask]  # (?, 28*28)
        for bin in range(32):
            match_bi = np.where(match_la == bin, 1, 1e-8)
            freq_bi = np.log(np.sum(match_bi, axis=0) / (match_la.shape[0]))
            likelihood[label, bin] = freq_bi

    print("finished likelihood cal")
    pred_y = np.ndarray((test_X.shape[0], 10))
    for d in progressBar(range(test_X.shape[0]), length=50):
        for label in range(10):
            for pix in range(28 * 28):
                pred_y[d, label] += likelihood[label, test_X[d, pix], pix]
        pred_y[d] += np.log(prior_train)

    return np.argmax(pred_y, axis=1), (pred_y / np.sum(pred_y, axis=1, keepdims=True)), likelihood


def infer(mode: Mode = Mode.DISCRETE):
    global train_X, train_y, test_X, test_y
    if mode == Mode.DISCRETE:

        # binning
        train_X = np.floor(train_X.astype(np.float32) / 8).astype(np.int32)
        test_X = np.floor(test_X.astype(np.float32) / 8).astype(np.int32)
        pred, posterior, likelihood = discrete(train_X, train_y, test_X, test_y)
        for i in range(posterior.shape[0]):
            print("Posterior (in log scale)")
            for l in range(10):
                print("{}: {}".format(l, float(posterior[i, l])))
            print("prediction: {}, ans: {}".format(int(pred[i]), int(test_y[i])))
            print()

        visualize_discrete(likelihood)
        print("error rate: {:.4f}".format(float(1 - np.mean(np.where(pred == test_y, 1, 0)))))

    elif mode == Mode.CONTINUOUS:
        pass


def visualize_discrete(likelihood: np.ndarray):
    
    likelihood = np.exp(likelihood) # original scale (10, 32, 28*28)
    black = np.sum(likelihood[:,16:, :], axis=1)
    white = np.sum(likelihood[:, :16, :], axis=1)
    img = np.where(black > white, 1, 0)
    for la in range(10):
        print(f"label {la}")
        for i in range(28 * 28):
            print(int(img[la, i]), end='')

            if i % 28 == 27:
                print()
        print()



if __name__ == "__main__":
    infer()
