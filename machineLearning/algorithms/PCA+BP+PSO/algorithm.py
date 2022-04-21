import numpy as np
import csv
# from reductionDim.PCA.algorithm import PCA
from algorithms import choose
from dataPretreatment import normalized

NormalizeType = {
    'benefit': 1,
    'cost': 2,
    'fixed': 3,
    'deviate': 4,
    'interval': 5,
    'de-interval': 6,
}


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
    def __init__(self, pwd):
        self.listOfFam = []
        self.drillVelocity = []
        # T: training, E: test
        self.trainingSet, self.testSet = self.csv_to_array(pwd)
        self.tGroups, self.tFeatures = self.trainingSet.shape
        self.eGroups, self.eFeatures = self.testSet.shape
        self.normalize()

    def csv_to_array(self, pwd):
        with open(pwd, encoding="utf-8-sig") as c:
            drillData = list(csv.reader(c))
            featureName = drillData.pop(0)
            methodName = drillData.pop(0)
        trainingSet, testSet = [], []

        # pop ZS from every line
        for i in drillData:
            self.drillVelocity.append(i.pop(-1))

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
            self.listOfFam.append(
                FeatureAndMethod.Struct(featureName[j],
                                        method,
                                        interval,
                                        j,
                                        averageOfCol,)
            )
            for row in emptyRow:
                drillData[row][j] = averageOfCol

        # choose n=1/4 datas for test, m=3/4 datas for training
        m, n = choose.random_choice_m_in_n(
            3 * int(len(drillData) / 4),
            len(drillData)
        )

        while n:
            testSet.append(drillData.pop(n.pop(-1)))
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
            ).reshape(self.tGroups,)
            self.testSet[:, fam.serial] = normalized.normalize(
                self.testSet[:, fam.serial].reshape(self.eGroups, 1),
                fam.method,
                fam.interval
            ).reshape(self.eGroups,)

    def decentralization(self, fam):
        self.trainingSet[:, fam.serial] -= fam.average
        self.testSet[:, fam.serial] -= fam.average
        if fam.method == NormalizeType['fixed'] or fam.method == NormalizeType['deviate']:
            fam.interval[0] -= fam.average
        elif fam.method == NormalizeType['interval'] or fam.method == NormalizeType['de-interval']:
            fam.interval[0] -= fam.average
            fam.interval[1] -= fam.average


algo = Algorithm('D:/Documents/gPaper/drillData.csv')
print('---')


