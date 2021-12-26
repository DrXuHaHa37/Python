import numpy as np
import matplotlib.pyplot as plt
import csv

import sys
sys.path.append("../../")
from dataPretreatment import normalized as nmlz

class pca():
	def __init__(self):
		dataSetPwd = "../../dataSet/drillData_CSV"
		self.dataType = 'csv'
		self.allFeatures = nmlz.putDatasTogether(dataSetPwd, self.dataType)
	def getinfo(self):
		print(self.allFeatures)
		print(len(self.allFeatures))

conclusion = pca()
conclusion.getinfo()