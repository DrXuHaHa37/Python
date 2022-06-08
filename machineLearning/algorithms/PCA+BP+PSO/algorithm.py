import numpy as np  # 1.20.3
import csv  #
import matplotlib.pyplot as plt  # 3.4.3
from reductionDim.PCA.algorithm import PCA_V_2
from algorithms import choose
from dataPretreatment import normalized
from algorithms.NN import BP

NormalizeType = {
    'zs': -1,
    'benefit': 1,
    'cost': 2,
    'fixed': 3,
    'deviate': 4,
    'interval': 5,
    'de-interval': 6,
}

CsvFilePwd = 'D:/Documents/gPaper/drillData.csv'
AccumulateVariance = 0.95
# 0.90 3
# 0.95 5
# 0.98 8


class FeatureAndMethod(object):
    # FAM = FeatureAndMethod
    # XXX = FAM.new_record(feature, method
    class Struct(object):
        def __init__(self, feature, method, interval, serial, average):
            self.feature = feature
            self.method = method
            self.interval = interval
            self.serial = serial
            self.average = average

    def new_record(self, feature, method):
        return self.Struct(feature, method)


class Algorithm:
    def __init__(self, pwd, accumulateV, generation):
        self.listOfFam = []
        self.drillVelocity_T = []
        self.drillVelocity_E = []
        # T: training, E: test
        self.trainingSet, self.testSet = self.csv_to_array(pwd)
        self.tGroups, self.tFeatures = self.trainingSet.shape
        self.eGroups, self.eFeatures = self.testSet.shape
        self.normalize()
        if self.eFeatures > 5:
            self.projectionMtx = PCA_V_2(self.trainingSet, accumulateV).projectionMtx
            self.lowDimT = np.dot(self.trainingSet, self.projectionMtx)
            self.lowDimE = np.dot(self.testSet, self.projectionMtx)
        else:
            self.lowDimT = self.trainingSet
            self.lowDimE = self.testSet
        self.pGroups, self.pFeatures = self.lowDimT.shape

        # build BP NN, save the pkl file in 'algorithms/NN/matrixT_training.pkl'
        self.hidden = [self.pFeatures + 10, self.pFeatures + 10, self.pFeatures + 10, self.pFeatures + 10]
        self.learnRate = 0.0001
        bp = BP.BP(self.lowDimT, self.drillVelocity_T, generation, self.learnRate, self.hidden)

        # test BP accurate
        bp.test_accuracy(self.lowDimE, self.drillVelocity_E)

    def csv_to_array(self, pwd):
        # pop features' name & method vector
        with open(pwd, encoding="utf-8-sig") as c:
            drillData = list(csv.reader(c))
            featureName = drillData.pop(0)
            methodName = drillData.pop(0)
        trainingSet, testSet = [], []

        # let empty elements' values = (average of effective of that column)
        for j in range(len(drillData[0])):
            emptyRow = []
            effectiveCount = []
            for i in range(len(drillData)):
                if drillData[i][j]:
                    drillData[i][j] = float(drillData[i][j])
                    effectiveCount.append(drillData[i][j])
                else:
                    emptyRow.append(j)
                    drillData[i][j] = 0.0
            averageOfCol = sum(effectiveCount) / len(effectiveCount)
            method, interval = self.depart_method_string(methodName[j])
            for row in emptyRow:
                drillData[row][j] = averageOfCol
            if method != NormalizeType['zs']:
                self.listOfFam.append(
                    FeatureAndMethod.Struct(featureName[j],
                                            method,
                                            interval,
                                            j,
                                            averageOfCol, )
                )
            else:
                for i in range(len(drillData)):
                    self.drillVelocity_T.append(drillData[i].pop())

        # choose n=1/4 datas for test, m=3/4 datas for training
        # exchange m & n
        m, n = choose.random_choice_m_in_n(
            3 * int(len(drillData) / 4),
            len(drillData)
        )

        while n:
            index = n.pop(-1)
            testSet.append(drillData.pop(index))
            self.drillVelocity_E.append(self.drillVelocity_T.pop(index))
        trainingSet = drillData
        return np.array(trainingSet), np.array(testSet)

    # depart row 2 in csv
    def depart_method_string(self, methodStr):
        issue = list(map(float, methodStr.split(',')))
        method = int(issue.pop(0))
        return method, issue

    # include decentralization & normalized
    def normalize(self):
        for fam in self.listOfFam:
            self.decentralization(fam)
            self.trainingSet[:, fam.serial] = normalized.normalize(
                self.trainingSet[:, fam.serial].reshape(self.tGroups, 1),
                fam.method,
                fam.interval
            ).reshape(self.tGroups, )
            self.testSet[:, fam.serial] = normalized.normalize(
                self.testSet[:, fam.serial].reshape(self.eGroups, 1),
                fam.method,
                fam.interval
            ).reshape(self.eGroups, )

    def decentralization(self, fam):
        self.trainingSet[:, fam.serial] -= fam.average
        self.testSet[:, fam.serial] -= fam.average
        if fam.method == NormalizeType['fixed'] or fam.method == NormalizeType['deviate']:
            fam.interval[0] -= fam.average
        elif fam.method == NormalizeType['interval'] or fam.method == NormalizeType['de-interval']:
            fam.interval[0] -= fam.average
            fam.interval[1] -= fam.average


algo = Algorithm(CsvFilePwd, AccumulateVariance, 5000)
