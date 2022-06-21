from algorithms.NN import NN
from algorithms import choose
from algorithms import optimize as opt
from reductionDim.PCA.algorithm import PCA_V_2
from algorithms.functions import csv_operate
from algorithms import draw
import numpy as np
import csv
import sys
sys.path.append('../')

NormalizeType = {
    'zs': -1,
    'benefit': 1,
    'cost': 2,
    'fixed': 3,
    'deviate': 4,
    'interval': 5,
    'de-interval': 6,
}


# 对原始数据进行处理, 将字符型属性转换为数字, 并返回pwd, 供[数据预处理]使用
class ProcessExcelFile:
    def __init__(self, readPwd, savePwd, featureName):
        self.drillData = csv_operate().readByPwd(readPwd)
        self.version_1(savePwd, featureName)

    def version_1(self, savePwd, featureName):
        self.make_dictionary_for_special_feature(featureName)
        csv_operate().saveByPwd(savePwd, self.drillData)

    def make_dictionary_for_special_feature(self, featureName):
        theDic = dict()
        featureSerial = 0
        theFeatureIndex = self.drillData[0].index(featureName)
        for tp in range(1, len(self.drillData)):
            feature = self.drillData[tp][theFeatureIndex]
            try:
                print(theDic[feature])
            except:
                featureSerial += 1
                theDic[feature] = featureSerial
            self.drillData[tp].insert(theFeatureIndex + 1, theDic[feature])


class Algorithm:
    def __init__(self, dataPwd: str, accumulateV: list, generations: int):
        self.accumulate = accumulateV
        self.generations = generations
        self.predictDict = dict()

        # 先对数据进行预处理
        dpp = self.DataPreProcessing(dataPwd)

        self.drillData, self.vpc = dpp.drillData, dpp.vpc

        # T 训练集  list    vT 训练集对应钻速
        # E 测试集合 list   vE 测试集对应钻速
        self.T, self.E, self.vT, self.vE = self.depart_training_test()
        self.count = 1
        self.limit = 10
        while self.count < 6:
            self.prepare()
            self.count += 1

    # 区分训练集和测试集
    def depart_training_test(self):
        # 降至不同维度
        trainingSet = []
        testSet = [[] for x in range(len(self.accumulate))]
        vpc_T = self.vpc.copy()
        vpc_E = []

        # 针对多种方差贡献率生成不同维度的数据
        for av in self.accumulate:
            trainingSet.append(list(map(list, (np.dot(self.drillData, PCA_V_2(self.drillData, av).projectionMtx)))))

        # 提取训练集(T)和测试集(E)
        # choose n=1/4 datas for test, m=3/4 datas for training
        # exchange m & n
        m, n = choose.random_choice_m_in_n(
            self.drillData.shape[0]
        )
        firstLoop = True
        for no in range(len(trainingSet)):
            nn = n.copy()
            while nn:
                index = nn.pop(-1)
                testSet[no].append(trainingSet[no].pop(index))
                if firstLoop:
                    vpc_E.append(vpc_T.pop(index))
            firstLoop = False

        # trainingSet 和 testSet可能是多维的 list传入
        return self.normalized(trainingSet), self.normalized(testSet), vpc_T, vpc_E

    # 数据预处理
    class DataPreProcessing:
        def __init__(self, pwd):
            # 保存钻参
            self.drillData = []
            # 保存钻速
            self.vpc = []
            # 开始处理数据
            self.get_datas_from_csv(pwd)

        # 将csv文件中的数据读到list中
        def get_datas_from_csv(self, pwd: str):
            with open(pwd, encoding="utf-8-sig") as c:
                self.drillData = list(csv.reader(c))
            # 对数据进行去中心化
            self.datas_normalized()

        # 去中心化
        def datas_normalized(self):
            featureName = self.drillData.pop(0)

            vpcIndex = featureName.index('ZS')

            rows = len(self.drillData)

            # 将csv读出的数据转化为float类型, 并弹出钻速
            for i in range(rows):
                self.drillData[i] = list(map(float, self.drillData[i]))
                self.vpc.append(self.drillData[i].pop(vpcIndex))

            # 将钻速归一化
            tVpc = np.array(self.vpc)
            self.vpc = list((tVpc - np.min(tVpc)) / (np.max(tVpc) - np.min(tVpc)))

            drillData_Array = np.array(self.drillData)
            for j in range(drillData_Array.shape[1]):
                # 归一化
                theMax, theMin = np.max(drillData_Array[:, j]), np.min(drillData_Array[:, j])
                drillData_Array[:, j] = (drillData_Array[:, j] - theMin) / (theMax - theMin)
                # 去中心化
                avg = np.average(drillData_Array[:, j])
                drillData_Array[:, j] = drillData_Array[:, j] - avg

            self.drillData = drillData_Array

    # 归一化
    def normalized(self, arr: np.array):
        tempArr = []
        for subArr in arr:
            subArr = np.array(subArr)
            for j in range(subArr.shape[1]):
                theMin = np.min(subArr[:, j])
                subArr[:, j] = (subArr[:, j] - theMin) / (np.max(subArr[:, j]) - theMin)
            tempArr.append(subArr)
        return tempArr

    # 去中心化
    def de_centered(self, arr: np.array):
        tempArr = []
        for subArr in arr:
            subArr = np.array(subArr)
            for j in range(subArr.shape[1]):
                avg = np.average(subArr[:, j])
                subArr[:, j] = subArr[:, j] - avg
            tempArr.append(subArr)
        return tempArr

    # 适应度函数
    def fitness_function(self):
        pass

    def prepare(self):
        subCount = 3

        while subCount < 10:
            hiddenLayer = [subCount for x in range(self.count)]
            hiddenLayer.append(1)
            self.training(hiddenLayer)
            subCount += 1

    # 训练 可以循环不同的算法
    def training(self, hD):
        predict = []
        algoBase = opt.ALGO(self.generations, 7)
        algoBase.hiddenLayer = hD
        optim_algorithms_list = [
            opt.PSO(algoBase),
            # opt.ACO,
            # opt.GA,
        ]
        # 当数据集不同维度时
        for subDataSet in self.T:
            dataset = np.array(subDataSet)
            dim = dataset.shape[1]

            # 当算法不同时
            for algo in optim_algorithms_list:
                algoName = str(algo) + '-' + str(dim)
                # 算法返回最优
                wbVector = algo.algorithm(dataset, self.vT)
                self.predictDict[algoName] = self.test(wbVector, algoBase, dim)

        self.check_dict_and_draw(algoBase.hiddenLayer)

    # 测试集测试训练结果
    def test(self, wb, algoInfo, dim):
        predict = []
        returnData = []
        for subDataSet in self.E:

            # 初始化数据集格式 神经网络
            dataset = np.array(subDataSet)

            rows, columns = dataset.shape
            if columns != dim:
                continue

            returnData = [np.linspace(1, rows, rows)]
            nn = NN.NN(columns, algoInfo.hiddenLayer)

            subPredict = []
            if nn.check_WB_hidden_number(list(wb)):
                for i in range(rows):
                    subPredict.append(nn.calculate(dataset[i]))
            predict.append(subPredict)

            returnData.append(np.array(predict))

        return returnData

    def check_dict_and_draw(self, hD):
        self.predictDict['Sample'] = [np.linspace(1, len(self.vE), len(self.vE)), np.array(self.vE)]
        for i in self.predictDict:
            self.predictDict[i][1].resize(self.predictDict[i][0].shape)

        title = '-'.join(list(map(str, hD)))

        draw.draw_multi_lines(self.predictDict, title)


# start

dataPwd = 'D:/Documents/drillData/readyToUse/'
# oldName = 'drillData_v4_before.csv'
# newName = 'drillData_v4.csv'
# ProcessExcelFile(orgPwd + oldName, orgPwd + newName, 'YXMS')

alg = Algorithm(dataPwd + 'drillData_v4.csv', [0.8, 0.95], 500)
