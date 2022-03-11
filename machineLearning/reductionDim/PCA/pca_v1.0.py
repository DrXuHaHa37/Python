import numpy as np
import matplotlib.pyplot as plt
import csv

import sys

sys.path.append("../../")

from dataPretreatment import normalized as nmlz


class datas():
    def __init__(self):
        dataSetPwd = "../../dataSet/drillData_CSV"
        self.dataType = 'csv'
        self.allFeatures = nmlz.get_features_from_datasets(dataSetPwd, self.dataType)

