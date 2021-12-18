import numpy as np
import matplotlib.pyplot as plt
import csv

import sys
sys.path.append("..")
from dataPretreatment import normalized as nmlz

dataSetPwd = "../dataSet/testData.csv"

matrix = []
with open(dataSetPwd) as dataFile:
	file_csv = csv.reader(dataFile)
	for row in range(1,len(file_csv)):
		matrix.append(row)

mtx = np.asarray(matrix)
print(mtx)



# arr1 = np.asarray(a,float)
# print(nmlz.normalize(arr1,3,[2,4,3]))
