import numpy as np
import matplotlib.pyplot as plt
import csv

import sys
<<<<<<< HEAD
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
=======

sys.path.append("../../")

from dataPretreatment import normalized as nmlz


class datas():
    def __init__(self):
        dataSetPwd = "../../dataSet/drillData_CSV"
        self.dataType = 'csv'
        self.allFeatures = nmlz.get_features_from_datasets(dataSetPwd, self.dataType)


newdata = datas()
>>>>>>> 7c6ebb69e1a4649314f6ea0c4646d946143390b9
