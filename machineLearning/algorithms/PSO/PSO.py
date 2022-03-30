import numpy as np
import sys
sys.path.append("../")
from draw import draw_convergence
import format
from functions import UNIMODAL

pLimit = [(-2, 2), (-2, 2)]
vLimit = [(-0.4, 0.4), (-0.4, 0.4)]
w = 0.8
c = (0.49, 1.49)
maxIter = 100
group = 5
modal = 'UNIMODAL'
modal_k = {
    'sphere': 1,
    'schwefel_problem_2_22': 2,
    'schwefel_problem_1_2': 3,
    'schwefel_problem_2_21': 4,
    'rosenBrock': 5, # 完全有问题
    'step': 6, # 收敛慢
    'quartic': 7 # 局部
}


def func(arr):
    return functions.UNIMODAL(arr).f2


class PSO:
    def __init__(self, ITER, GROUP, LIMIT_P, LIMIT_V, MODAL, MODAL_K, W, C):
        self.iter = ITER
        self.group = GROUP
        self.limit_P = LIMIT_P
        self.limit_V = LIMIT_V
        self.pBest = [float('inf') for x in range(self.group)]
        self.gBest = [float('inf')]
        self.w = W
        self.c = C
        self.modal = MODAL
        self.modalK = MODAL_K

        # init position & velocity
        self.dim = len(LIMIT_P)
        self.position = np.random.rand(self.group, self.dim)
        self.velocity = np.random.rand(self.group, self.dim)
        self.init_matrix(self.position, self.limit_P)
        self.init_matrix(self.velocity, self.limit_V)

    def init_matrix(self, mtx, limit):
        for dim in range(self.dim):
            mtx[:, dim] = mtx[:, dim] * (limit[dim][1] - limit[dim][0]) + limit[dim][0]

    def limit_column(self, arr, column, model='p', isInit=False):
        up, down = (self.limit_P[column][1], self.limit_P[column][0]) if (model == 'p') \
                    else (self.limit_V[column][1], self.limit_V[column][0])
        if isInit:
            return arr * (up - down) + down
        else:
            for value in range(len(arr)):
                if arr[value] > up:
                    arr[value] = up
                elif arr[value] < down:
                    arr[value] = down
            return arr

    def algorithm(self):
        for Iter in range(self.iter):
            for dim in range(self.dim):
                self.position[:, dim] = format.format_column(self.limit_P[dim], self.position[:, dim])
                self.velocity[:, dim] = format.format_column(self.limit_V[dim], self.velocity[:, dim])
            for identity in range(self.group):
                tempF = self.function(self.position[identity])
                if tempF < self.pBest[identity]:
                    self.pBest[identity] = tempF
            self.gBest.append(min(self.pBest))
            for identity in range(self.group):
                self.velocity[identity] = self.w * self.velocity[identity] + \
                                          self.c[0] * np.random.random() * (self.pBest[identity] - self.position[identity]) + \
                                          self.c[1] * np.random.random() * (self.gBest[-1] - self.position[identity])
                self.position[identity] = self.position[identity] + self.velocity[identity]

            print("[第{}代] 全局最优值:{}\n".format(Iter, self.gBest[-1]))
        draw_convergence(self.gBest)

    def algorithm_v_2_0(self):
        newBestPoint = np.zeros(2,)
        for Iter in range(self.iter):
            for dim in range(self.dim):
                self.position[:, dim] = format.format_column(self.limit_P[dim], self.position[:, dim])
                self.velocity[:, dim] = format.format_column(self.limit_V[dim], self.velocity[:, dim])
            if self.modal == "UNIMODAL":
                fitness = UNIMODAL(self.position, self.modalK).returnValue

            for identity in range(self.group):
                if fitness[identity] < self.pBest[identity]:
                    self.pBest[identity] = fitness[identity]
                if fitness[identity] < self.gBest[-1]:
                    self.gBest.append(fitness[identity])
                    newBestPoint = self.position[identity]
                else:
                    self.gBest.append(self.gBest[-1])
            for identity in range(self.group):
                self.velocity[identity] = \
                    self.w * self.velocity[identity] + \
                    self.c[0] * np.random.random() * (self.pBest[identity] - self.position[identity]) + \
                    self.c[1] * np.random.random() * (self.gBest[-1] - self.position[identity])
                self.position[identity] = self.position[identity] + self.velocity[identity]
            print("[第{}代] 全局最优值:{}, 最优点:{}\n".format(Iter + 1, self.gBest[-1], newBestPoint))
        draw_convergence(self.gBest)


class NMOPSO(PSO):
    def __init__(self, ITER, GROUP, LIMIT_P, LIMIT_V, FUNC, W, C):
        super(NMOPSO, self).__init__(ITER, GROUP, LIMIT_P, LIMIT_V, FUNC, W, C)


pso = PSO(maxIter, group, pLimit, vLimit, modal, 7, w, c)
pso.algorithm_v_2_0()

