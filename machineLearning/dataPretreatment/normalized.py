import numpy as np
import os
import csv


# 数据归一化方法：
# 1.效益型 越大越好-----------	> 大的大
# 2.成本型 越小越好-----------	> 小的大
# 3.固定型 越接近某个值越好	---->
# 4.偏离型 约远离某个值越好----->
# 5.区间型 越靠近某个区间越好---	>


# 传入numpy.array
# 传入每个属性对应的归一化方式 一维向量
def normalize(arr, method, standard=None):
    columnCount = len(arr[0])
    rowCount = len(arr)

    if (standard is not None) and (len(standard) != columnCount):
        return "error: wrong standard length"

    for column in range(columnCount):
        currentColumn = arr[:, column]
        maxOfColumn = np.max(currentColumn)
        minOfColumn = np.min(currentColumn)

        for row in range(rowCount):
            if method[column] == 1:
                arr[row, column] = (arr[row, column] - minOfColumn) / (maxOfColumn - minOfColumn)
            elif method[column] == 2:
                arr[row, column] = (maxOfColumn - arr[row, column]) / (maxOfColumn - minOfColumn)
            elif method[column] == 3 and standard:
                arr[row, column] = (abs(standard[column] - arr[row, column])) / (minOfColumn - maxOfColumn) + 1
            elif method[column] == 4 and standard:
                arr[row, column] = (abs(arr[row, column] - standard[column]) / (maxOfColumn - minOfColumn))
            elif method[column] == 5:
                pass
            else:
                return "error: wrong normalize method input"
    return arr


# put datasets which the path ordered together, and ends with '.dataType'
def putDatasTogether(path, dataType):
    files = os.listdir(path)
    featureSet = set()
    for file in files:
        if not (file.endswith('.' + dataType) and file.startswith('drill')):
            files.pop(files.index(file))
    files.pop(files.index('README.md'))
    with open(path + "/attributes.py", encoding="utf-8") as attr:
        existAttr = len(attr.readlines())
    for file in files:
        with open(path + '/' + file, encoding="utf-8-sig") as f:
            csvFile = list(csv.reader(f))
            features = csvFile[0]
        for feature in features:
            feature = feature.split(' ')
            try:
                feature[1] = "\"" + feature[1] + "\""
            except:
                feature.append("\"\"")
            finally:
                feature[-1] = feature[-1] + '\n'
                featureSet.add(' = '.join(feature))
    featureSet = list(featureSet)
    featureSet.pop(featureSet.index(' = ""\n'))
    if len(featureSet) != existAttr:
        with open(path + "/attributes.py", "wt", encoding="utf-8-sig") as attr:
            attr.writelines(sorted(featureSet))
    return sorted(featureSet)
