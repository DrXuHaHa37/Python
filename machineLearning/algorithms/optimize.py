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


def nnFit(wbVector, nn, x):
    rows = x.shape[0]
    loopFit = np.array([])
    if nn.check_WB_hidden_number(wbVector):
        for i in range(rows):
            loopFit = np.append(loopFit, nn.calculate(x[i]))
    return loopFit

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


class GA(ALGO):
    # 目标函数 交叉率 变异率 代数 准确度
    def __init__(self, algo: ALGO, cross_rate, mutate_rate):
        super().__init__(algo.iter, algo.group)
        self.algoBase = algo
        self.serial = AlgoDict['GA']

        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate

    def __str__(self):
        return 'GA'

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

        # 保存训练历史中最好的适应度值
        gBestP = np.zeros(self.group)
        gBest = [float('inf')]

        for _ in range(self.iter):
            # 保存本轮循环每个个体适应度
            fit = np.array([])
            for identity in range(self.group):
                # 每一行数据求出nn输出 向量
                currentLoopFit = nnFit(list(groups[identity]), nn, x)
                # 预测值向量和真实值向量做适应度求解
                currentFitness = fitness(yReal, currentLoopFit)

                # 本轮循环每个个体的适应度更新
                fit = np.append(fit, currentFitness)
                # 如果本次循环 个体适应度比历史个体适应度好

            if np.min(fit) < gBest[-1]:
                gBestP = groups[np.argmin(fit)]
                gBest.append(np.min(fit))
            else:
                gBest.append(gBest[-1])

            # 遗传算法训练
            groups = self.choose(fit, groups)
            groupCopy = groups.copy()
            for parent in range(groups.shape[0]):
                child = self.cross(groups[parent], groupCopy)
                child = self.mutate(child)
                groups[parent] = child

        return gBestP



    # 选择
    def choose(self, fit, groups):
        index = np.random.choice(np.arange(self.group), size=self.group, p=fit / fit.sum())
        return groups[index]

    # 交叉
    def cross(self, parent, group):
        if np.random.rand() < self.cross_rate:
            # 选择一个个体序号
            idx = np.random.randint(0, self.group, size=1)
            # 将父代和该序号进行交叉
            beta = np.random.randn()
            parent = beta * parent + (1 - beta) * group[idx]
        return parent

    # 变异
    def mutate(self, child):
        for i in range(len(child)):
            if np.random.rand() < self.mutate_rate:
                child[i] += np.random.randn()
        return child


