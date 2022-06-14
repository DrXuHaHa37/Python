import matplotlib.pyplot as plt
import csv

import sys
from sympy import *
import numpy as np
import math


class PCA:
    def __init__(self, dataPwd):
        self.dataPwd = dataPwd
        self.attributes = []
        self.matrix = self.get_data_from_excel(dataPwd)
        self.pca_v_1_0()

    def get_data_from_excel(self, dataPwd):
        with open(dataPwd, encoding="utf-8-sig") as f:
            drillData = list(csv.reader(f))
            self.attributes = drillData.pop(0)
        for i in range(len(drillData)):
            for j in range(len(drillData[0])):
                if drillData[i][j]:
                    drillData[i][j] = float(drillData[i][j])
                else:
                    drillData[i][j] = 0.0
        return np.array(drillData).T

    def pca_v_1_0(self):
        avgOfCol = []
        attrs, groups = self.matrix.shape

        # initiate: every dim minus the average of the dim
        for attr in range(attrs):
            avgOfCol.append(np.sum(self.matrix[attr])/groups)
            self.matrix[attr] -= avgOfCol[attr]
        covMtx = np.cov(self.matrix)
        eigenValue, featureVector = np.linalg.eig(covMtx)
        eigenContribution = eigenValue / np.sum(eigenValue)
        for k in range(len(list(eigenContribution))):
            if eigenContribution[k] < 0.01:
                break
            with open(self.dataPwd,  "wt", encoding="utf-8-sig", newline="") as f:
                f_writer = csv.writer(f)
                for i in list(map(list, np.dot(self.matrix.T, featureVector[:, :k]))):
                    newData = list(list(map(str, i)))
                    f_writer.writerows(newData)


class PCA_V_2:
    def __init__(self, mtx: np.array, accVariance):
        # mtx = groups * attributes
        self.matrix = mtx
        self.projectionMtx = self.algorithm_projectionMtx(accVariance)

    def algorithm_projectionMtx(self, accV):
        covMtx = np.dot(self.matrix.T, self.matrix)
        eigenValue, featureVector = np.linalg.eig(covMtx)
        accumulate = 0
        columnNo = -1
        totalVariance = np.sum(eigenValue)
        for value in list(eigenValue):
            accumulate += value
            columnNo += 1
            if accumulate / totalVariance > accV:
                break
        return featureVector[:, :columnNo + 1]


a = [[2, -2, 0], [-2, 1, -2], [0, -2, 0]]

pca = PCA_V_2(np.array(a), 0.9)
