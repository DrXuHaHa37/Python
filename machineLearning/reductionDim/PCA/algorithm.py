import numpy as np
import matplotlib.pyplot as plt
import csv

import sys
from sympy import *
import numpy as np
import math


class PCA:
    def __init__(self, dataPwd):
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
        with open("D:\Documents\gPaper\PCA.csv",  "wt", encoding="utf-8-sig", newline="") as f:
            f_writer = csv.writer(f)
            newData = list(list(map(str, i))
                                    for i in list(map(list, np.dot(self.matrix.T, featureVector[:, :k]))))
            f_writer.writerows(newData)


class PCA_V_2:
    def __init__(self, mtx):
        self.matrix = mtx


dataSetPwd = "D:/Documents/gPaper/drillData.csv"
pca = PCA(dataSetPwd)
