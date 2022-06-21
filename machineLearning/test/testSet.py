import numpy as np

class Algo:
    def __init__(self, iters, groups):
        self.iter = iters
        self.group = groups


class PSO(Algo):
    def __init__(self, w, c, algo: Algo):
        self.iter = algo.iter
        self.group = algo.group
        self.w = w
        self.c = c

    def algorithm(self):
        pass


def normalized(arr: np.array):
    for j in range(arr.shape[1]):
        theMin = np.min(arr[:, j])
        arr[:, j] = (arr[:, j] - theMin) / (np.max(arr[:, j]) - theMin)
    return arr


a = np.array([[1, 7], [9, 4]])
print(normalized(a))

