import numpy as np
from algorithms.NN import NN
from algorithms import draw


AlgoDict = {
    'PSO': 1,
    'ACO': 2,
    'GA': 3,
}

INF = np.inf


def fitness(real, predict):
    lgh = len(predict)
    pred = np.array(predict)
    loss = np.sum((real - pred) ** 2) / lgh
    return loss


# 算法输入: 最大代数, 种群数, 输入, 真实值
# 返回神经网络参数向量 np.array
class ALGO:
    def __init__(self, iters, groups):
        self.iter = iters
        self.group = groups
        self.hiddenLayer = [7, 7, 1]


class PSO(ALGO):
    def __init__(self, algo: ALGO, w: int = 0.25, c: tuple = (2, 2)):
        super().__init__(algo.iter, algo.group)
        self.algoBase = algo
        self.w = w
        self.c = c
        self.serial = AlgoDict['PSO']

    def __str__(self):
        return 'PSO'

    def algorithm(self, x: np.array, yReal: list):
        # 初始化神经网络结构, 输入层-隐层-输出层 层数&神经元个数
        rows, dim = x.shape
        yReal = np.array(yReal)
        nn = NN.NN(dim, self.algoBase.hiddenLayer)

        # 粒子群算法初始化
        # 计算WB向量维度
        bStandard = sum(self.algoBase.hiddenLayer)
        wStandard = dim * self.algoBase.hiddenLayer[0]
        for i in range(len(self.algoBase.hiddenLayer) - 1):
            wStandard += self.algoBase.hiddenLayer[i] * self.algoBase.hiddenLayer[i + 1]
        wbLength = wStandard + bStandard

        # 初始化 并判断是否合法
        groups = np.random.randn(self.group, wbLength)
        v = np.random.rand(self.group, wbLength)
        # 保存每个个体历史上最好的值
        pBest = [float('inf') for k in range(self.group)]
        # 保存训练历史中最好的适应度值
        gBest = [float('inf')]

        pBestP = np.zeros((self.group, wbLength))
        gBestP = np.zeros(self.group)

        # 循环代数
        for iterNo in range(self.iter):
            # 循环个体
            for identity in range(self.group):
                currentLoopFit = []
                # 神经网络权值阈值初始化为identity值
                if nn.check_WB_hidden_number(list(groups[identity])):
                    # 输入矩阵按行求值 求出该个体本次循环的适应度
                    for i in range(rows):
                        currentLoopFit.append(nn.calculate(x[i]))
                    currentFitness = fitness(yReal, currentLoopFit)

                    # 如果本次循环 个体适应度比历史个体适应度好
                    if currentFitness < pBest[identity]:
                        pBest[identity] = currentFitness
                        pBestP[identity] = groups[identity]

            # 若当前循环出现了新的比gBest更小的适应度
            newBestFitness = min(pBest)
            if newBestFitness < gBest[-1]:
                gBest.append(newBestFitness)
                gBestP = groups[pBest.index(newBestFitness)]
            else:
                gBest.append(gBest[-1])

            # 更新速度和位置
            for identity in range(self.group):
                v[identity] = \
                    self.w * v[identity] + \
                    self.c[0] * np.random.rand() * (pBestP[identity] - groups[identity]) + \
                    self.c[1] * np.random.rand() * (gBestP[-1] - groups[identity])
                groups[identity] = groups[identity] + v[identity]
        # draw.draw_with_text(np.linspace(1, self.iter + 1, self.iter + 1), np.array(gBest))

        return gBestP


class GA:
    def __init__(self, algo: ALGO):
        super().__init__(algo.iter, algo.group)
        self.algoBase = algo
        self.serial = AlgoDict['GA']

    def __str__(self):
        return 'GA'

    def algorithm(self, x: np.array, yReal: list):
        # 初始化神经网络结构, 输入层-隐层-输出层 层数&神经元个数
        rows, dim = x.shape
        yReal = np.array(yReal)
        nn = NN.NN(dim, self.algoBase.hiddenLayer)

    # 选择
    def choose(self):
        pass

    # 交叉
    def exchange(self):
        pass

    # 变异
    def mutate(self):
        pass


