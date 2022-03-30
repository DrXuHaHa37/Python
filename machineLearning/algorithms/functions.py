import numpy as np
from matplotlib import pyplot as plt

MATH_PI = np.pi
MATH_E = np.e


class UNIMODAL:
    # input the matrix of identity, and get the fitness vector
    def __init__(self, identityMatrix, k=0):
        self.X = np.asarray(identityMatrix)
        self.identity, self.dim = np.shape(self.X)
        self.result = np.zeros((self.identity, 7))
        self.index = {
            1: self.sphere(),
            2: self.schwefel_problem_2_22(),
            3: self.schwefel_problem_1_2(),
            4: self.schwefel_problem_2_21(),
            5: self.rosenBrock(),
            6: self.step(),
            7: self.quartic()
        }
        if k > 0:
            self.returnValue = self.result[:, k-1]

    def sphere(self):
        for d in range(self.dim):
            self.result[:, 0] += self.X[:, d] ** 2

    def schwefel_problem_2_22(self):
        self.result[:, 1] += 1
        for d in range(self.dim):
            self.result[:, 1] *= abs(self.X[:, d])
        for d in range(self.dim):
            self.result[:, 1] += abs(self.X[:, d])

    def schwefel_problem_1_2(self):
        for d in range(self.dim):
            temp = np.zeros((self.identity, ))
            for j in range(d+1):
                temp += self.X[:, j]
            self.result[:, 2] += temp ** 2

    def schwefel_problem_2_21(self):
        for i in range(self.identity):
            self.result[i, 3] = np.max(abs(self.X[i]))

    def rosenBrock(self):
        for d in range(self.dim - 1):
            self.result[:, 4] += (100*(self.X[:, d+1] - self.X[:, d] ** 2) ** 2+(self.X[:, d] - 1) ** 2)

    def step(self):
        for d in range(self.dim):
            self.result[:, 5] += (self.X[:, d] + 0.5) ** 2

    def quartic(self):
        for d in range(self.dim):
            self.result[:, 6] += (d+1) * (self.X[:, d] ** 4)
        rand = np.random.rand(self.identity, )
        self.result[:, 6] += rand


class MULTIMODAL:
    # input the matrix of identity, and get the fitness vector
    def __init__(self, identityMatrix, k):
        self.X = np.asarray(identityMatrix)
        self.identity, self.dim = np.shape(self.X)
        self.result = np.zeros((self.identity, 9))
        self.index = {
            1: self.schwefel_problem_2_26(),
            2: self.rastrigin(),
            3: self.ackley(),
            4: self.griewank(),
        }
        self.returnValue = self.result[:, k - 1]

    def schwefel_problem_2_26(self):
        for d in range(self.dim):
            self.result[:, 0] += -self.X[:, d] * np.sin(np.sqrt(abs(self.X[:, d])))

    def rastrigin(self):
        for d in range(self.dim):
            self.result[:, 1] += (self.X[:, d] ** 2 - 10 * np.cos(2 * MATH_PI * self.X[:, d]) + 10)

    def ackley(self):
        self.result[:, 2] += (20 + MATH_E)
        part = np.zeros((self.identity, 2))
        for d in range(self.dim):
            # 0 sqrt part
            part[:, 0] += self.X[:, d] ** 2
            # 1 cos part
            part[:, 1] += np.cos(2 * MATH_PI * self.X[:, d])
        part /= self.dim
        self.result[:, 2] += -20 * np.exp(-0.2 * np.sqrt(part[:, 0])) - np.exp(part[:, 1])

    def griewank(self):
        self.result[:, 3] += 1
        part = np.zeros((2, self.identity))
        part[1] += 1
        for d in range(self.dim):
            part[0] += self.X[:, d] ** 2
            part[1] *= np.cos(self.X[:, d] / np.sqrt(d+1))
        part[0] /= 4000
        self.result[:, 3] += (part[0] - part[1])

    def penalized(self):
        part = np.zeros((2, self.identity))
        for d in range(self.dim-1):
            part[0] += \
                (self.yi_for_penalized(self.X[:, d]) - 1) ** 2 * \
                (1 + 10 * (np.sin(MATH_PI * self.yi_for_penalized(self.X[:, d+1]))) ** 2)
            part[1] += self.u_for_penalized(self.X[:, d])
        part[1] += self.u_for_penalized(self.X[:, self.dim-1])
        self.result[:, 4] += part[0] + 10 * np.sin(MATH_PI * self.yi_for_penalized(self.X[:, 0])) + \
                             (self.yi_for_penalized(self.X[:, -1]) - 1) ** 2
        self.result[:, 4] *= (MATH_PI/self.dim)
        self.result[:, 4] += part[1]

    def yi_for_penalized(self, xi):
        # xi: the ith dim in matrix
        return 1 + ((xi + 1)/4)

    def u_for_penalized(self, xi):
        # xi: the ith dim in matrix
        a, k, m = 10, 100, 4
        u = np.zeros(self.identity,)
        for i in range(self.identity):
            if xi[i] > a:
                u[i] = k * (xi[i] - a) ** m
            elif xi[i] < -a:
                u[i] = k * (-xi[i] - a) ** m
        return u





