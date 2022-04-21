import numpy as np
import sys
sys.path.append("../")
from draw import draw_convergence
import format
from functions import UNIMODAL
import choose

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
    def __init__(self, ITER, GROUP, LIMIT_P, LIMIT_V, MODAL, MODAL_K, W, C):
        super(NMOPSO, self).__init__(ITER, GROUP, LIMIT_P, LIMIT_V, MODAL, MODAL_K, W, C)
        # self.sigmaShare = SIGMA
        # self.alpha = ALPHA

        self.outerCapacity = int(self.group / 2) if self.group > 3 else 2
        # self.beta = BETA
        self.outerDocument = self.position[0].reshape(1, self.dim)
        self.fitnessOfSwarm = [float('inf')]

    def algorithm(self):
        for Iter in range(self.iter):
            for dim in range(self.dim):
                self.position[:, dim] = format.format_column(self.limit_P[dim], self.position[:, dim])
                self.velocity[:, dim] = format.format_column(self.limit_V[dim], self.velocity[:, dim])
            if self.modal == "UNIMODAL":
                fitness = UNIMODAL(self.position, self.modalK).returnValue
                swarmFitness = UNIMODAL(self.outerDocument, self.modalK).returnValue
            else:
                fitness = []
                swarmFitness = []

            for identity in range(self.group):
                if fitness[identity] < self.pBest[identity]:
                    np.append(self.outerDocument, self.position[identity].reshape(1, self.dim), axis=0)
                    self.pBest[identity] = fitness[identity]

            outerDict = {}
            for identity in range(len(self.outerDocument)):
                outerDict[tuple(self.outerDocument[identity])] = swarmFitness[identity]
            self.gBest.append(outerDict[choose.roulette(outerDict)])

            for identity in range(self.group):
                self.velocity[identity] = \
                    self.w * self.velocity[identity] + \
                    self.c[0] * np.random.random() * (self.pBest[identity] - self.position[identity]) + \
                    self.c[1] * np.random.random() * (self.gBest[-1] - self.position[identity])
                self.position[identity] = self.position[identity] + self.velocity[identity]
            self.swarm(outerDict)
        draw_convergence(self.gBest)

    def swarm(self, dic):
        sortedDict = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        self.outerDocument = list(map(tuple, self.outerDocument))
        while len(self.outerDocument) > self.outerCapacity:
            self.outerDocument.pop(self.outerDocument.index(sortedDict.pop()[0]))
        self.outerDocument = np.array(self.outerDocument)
        self.logistic_chaos(0.7, 3, 2)

    def logistic_chaos(self, pCs, nCs, csMax):
        limit = list(map(list, self.limit_P.copy()))
        for d in range(self.dim):
            limit[d][0] = np.min(self.position[:, d])
            limit[d][1] = np.max(self.position[:, d])
            temp1 = limit[d][0] - 0.1 * abs(limit[d][0])
            temp2 = limit[d][1] + 0.1 * abs(limit[d][1])
            limit[d][0] = self.limit_P[d][0] if temp1 <= self.limit_P[d][0] else temp1
            limit[d][1] = self.limit_P[d][1] if temp2 >= self.limit_P[d][1] else temp2
        for h in range(nCs):
            randomIndex = np.random.randint(len(self.outerDocument))
            xH = self.outerDocument[randomIndex].copy()
            uJ = 0
            while uJ % 0.25 == 0:
                uJ = np.random.random()
            for v in range(csMax):
                fitnessOfOuter = UNIMODAL(self.outerDocument, self.modalK).returnValue
                uJ = (2 * ((v + csMax + 1)/csMax)) * uJ * (1-uJ)
                for d in range(self.dim):
                    t1 = min(self.limit_P[d][1], limit[d][1] - xH[d])
                    t2 = min(self.limit_P[d][1], xH[d] - limit[d][0])
                    cJ = t1 + t2
                    dJ = t2 * self.limit_P[d][1]
                    s = np.random.random()
                    xH[d] = xH[d] + cJ * uJ - dJ if s < pCs else xH[d]
                if self.modal == 'UNIMODAL':
                    fitnessOfNew = UNIMODAL(xH.reshape(1, self.dim), self.modalK).returnValue
                else:
                    fitnessOfNew = []
                for outer in range(len(self.outerDocument)):
                    if fitnessOfNew < fitnessOfOuter[outer]:
                        self.outerDocument[outer] = xH
                res = []
                for X in list(map(list, self.outerDocument)):
                    if X not in res:
                        res.append(X)
                self.outerDocument = np.array(res)


nmopso = PSO(maxIter, group, pLimit, vLimit, modal, 7, w, c)
nmopso.algorithm_v_2_0()

