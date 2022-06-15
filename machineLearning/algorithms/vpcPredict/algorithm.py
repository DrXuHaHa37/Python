from algorithms.NN import NN
from algorithms import choose
from algorithms import optimize as opt
from reductionDim.PCA.algorithm import PCA_V_2
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


class Algorithm:
    def __init__(self, orgPwd: str, accumulateV: list, generations: int):
        self.accumulate = accumulateV
        self.generations = generations

        # process the dataset by process_excel_file & DataPreProcessing
        dpp = self.DataPreProcessing(
                self.ProcessExcelFile(orgPwd).version_1(),
            )

        self.drillData, self.vpc = dpp.drillData, dpp.vpc

        # T 训练集  list    vT 训练集对应钻速
        # E 测试集合 list   vE 测试集对应钻速
        self.T, self.E, self.vT, self.vE = self.depart_training_test()
        self.training()

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

        return trainingSet, testSet, vpc_T, vpc_E

    # 对原始数据进行处理, 将字符型属性转换为数字, 并返回pwd, 供[数据预处理]使用
    class ProcessExcelFile:
        def __init__(self, orgPwd):
            self.pwd = orgPwd

        def version_1(self):
            return 'D:/Documents/Programing/Github/Python/machineLearning/dataSet/write_ready_to_use/drillData_v3.csv'

    # 数据预处理
    class DataPreProcessing:
        def __init__(self, pwd):
            self.drillData = []
            self.vpc = []

            self.get_datas_from_csv(pwd)

        # 将csv文件中的数据读到list中
        def get_datas_from_csv(self, pwd: str):
            with open(pwd, encoding="utf-8-sig") as c:
                self.drillData = list(csv.reader(c))
            self.datas_normalized()

        # 去中心化和归一化
        def datas_normalized(self):
            featureName = self.drillData.pop(0)

            vpcIndex = featureName.index('ZS')

            rows = len(self.drillData)

            # 将csv读出的数据转化为float类型, 并弹出钻速
            for i in range(rows):
                self.drillData[i] = list(map(float, self.drillData[i]))
                self.vpc.append(self.drillData[i].pop(vpcIndex))

            tVpc = np.array(self.vpc)
            self.vpc = list((tVpc - np.min(tVpc)) / (np.max(tVpc) - np.min(tVpc)))

            drillData_Array = np.array(self.drillData)
            temp_Array = drillData_Array.copy()

            self.drillData = temp_Array

    # 适应度函数
    def fitness_function(self):
        pass

    # 训练 可以循环不同的算法
    def training(self):
        predict = []
        algoBase = opt.ALGO(100, 5)
        optim_algorithms_list = [
            opt.PSO(algoBase),
            # opt.ACO,
            # opt.GA,
        ]
        # 当数据集是不同维度时
        for subDataSet in self.T:
            dataset = np.array(subDataSet)

            # 当算法不同时

            for algo in optim_algorithms_list:
                wbVector = algo.algorithm(dataset, self.vT)
                predict.append(self.test(wbVector, algoBase))

        predict = np.array(predict)
        print(predict)

    # 测试集测试训练结果
    def test(self, wb, algoInfo):
        predict = []
        for subDataSet in self.E:
            # 初始化数据集格式 神经网络
            dataset = np.array(subDataSet)
            rows, columns = dataset.shape
            nn = NN.NN(columns, algoInfo.hiddenLayer)

            subPredict = []
            if nn.check_WB_hidden_number(list(wb)):
                for i in range(rows):
                    subPredict.append(nn.calculate(i))
            predict.append(subPredict)
        return predict


# start
alg = Algorithm('dataSet/drillData_v3.csv', [0.3, 1], 100)
