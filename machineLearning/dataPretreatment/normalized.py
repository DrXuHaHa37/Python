import numpy as np
import os
import csv


# 数据归一化方法：
# 1.效益型 越大越好-----------	> 大的大
# 2.成本型 越小越好-----------	> 小的大
# 3.固定型 越接近某个值越好	---->
# 4.偏离型 约远离某个值越好----->
# 5.区间型 越靠近某个区间越好--->
# 6.偏离区间 约远离某个区间好---->
NormalizeType = {
    'benefit': 1,
    'cost': 2,
    'fixed': 3,
    'deviate': 4,
    'interval': 5,
    'de-interval': 6,
}


# 传入numpy.array
# 传入每个属性对应的归一化方式 一维向量
def normalize(arr, method, standard=None):
    rowCount, columnCount = arr.shape

    for column in range(columnCount):
        currentColumn = arr[:, column]
        maxOfColumn = np.max(currentColumn)
        minOfColumn = np.min(currentColumn)
        limit = (minOfColumn, maxOfColumn)
        k = get_K_for_normalize(method, limit, standard)

        if maxOfColumn == minOfColumn:
            pass
        elif method == NormalizeType['benefit']:
            arr[:, column] = (arr[:, column] - minOfColumn) / (maxOfColumn - minOfColumn)
        elif method == NormalizeType['cost']:
            arr[:, column] = (maxOfColumn - arr[:, column]) / (maxOfColumn - minOfColumn)
        elif method == NormalizeType['fixed'] or method == NormalizeType['interval']:
            for row in range(rowCount):
                if arr[row, column] <= standard[0]:
                    arr[row, column] = k * (arr[row, column] - standard[0]) + 1
                elif arr[row, column] >= standard[-1]:
                    arr[row, column] = -k * (arr[row, column] - standard[-1]) + 1
                else:
                    arr[row, column] = 1.

        elif method == NormalizeType['deviate'] or method == NormalizeType['de-interval']:
            for row in range(rowCount):
                if arr[row, column] <= standard[0]:
                    arr[row, column] = k * (arr[row, column] - standard[0])
                elif arr[row, column] >= standard[-1]:
                    arr[row, column] = -k * (arr[row, column] - standard[-1])
                else:
                    arr[row, column] = 0.
        else:
            return "wrong method were inputted!"
    return arr


def get_K_for_normalize(method, limit, standard):
    k = 0.0
    if method == NormalizeType['cost'] or method == NormalizeType['benefit']:
        if limit[-1] - limit[0] == 0:
            return 1
        elif method == NormalizeType['cost']:
            return 1 / (limit[0] - limit[-1])
        else:
            return 1 / (limit[-1] - limit[0])
    k1 = limit[-1] - standard[-1]
    k3 = limit[0] - standard[0]
    if limit[0] > standard[-1]:
        if method == NormalizeType['fixed'] or method == NormalizeType['interval']:
            k = 1 / k1
        elif method == NormalizeType['deviate'] or method == NormalizeType['de-interval']:
            k = 1 / (- k1)
    elif limit[-1] < standard[0]:
        if method == NormalizeType['fixed'] or method == NormalizeType['interval']:
            k = 1 / (- k3)
        elif method == NormalizeType['deviate'] or method == NormalizeType['de-interval']:
            k = 1 / k3
    elif abs(k1) < abs(k3) and standard[0] >= limit[0] and standard[-1] < limit[-1]:
        if method == NormalizeType['fixed']:
            k = 1 / (-k3)
        elif method == NormalizeType['deviate']:
            k = 1 / k3
    elif abs(k1) >= abs(k3) and standard[0] >= limit[0] and standard[-1] < limit[-1]:
        if method == NormalizeType['fixed'] or method == NormalizeType['interval']:
            k = 1 / k1
        elif method == NormalizeType['deviate'] or method == NormalizeType['de-interval']:
            k = 1 / (- k1)
    else:
        if standard[0] < limit[0]:
            if method == NormalizeType['interval']:
                k = 1 / k1
            elif method == NormalizeType['de-interval']:
                k = 1 / (- k1)
        elif standard[-1] > limit[-1]:
            if method == NormalizeType['interval']:
                k = 1 / (-k3)
            elif method == NormalizeType['de-interval']:
                k = 1 / k3
    return k


# put datasets which the path ordered together, and ends with '.dataType'
def get_features_from_datasets(path, dataType):
    files = os.listdir(path)
    featureDict = dict()
    featureFormula = []
    for file in files:
        if not (file.endswith('.' + dataType) and file.startswith('drill')):
            files.pop(files.index(file))
    with open(path + "/attributes.py", encoding="utf-8") as attr:
        existAttr = len(attr.readlines())
    for file in files:
        with open(path + '/' + file, encoding="utf-8-sig") as f:
            csvFile = list(csv.reader(f))
            features = csvFile[0]
        for feature in features:
            feature = feature.split(' ')
            featureDict[feature[0]] = '""\n'
            try:
                featureDict[feature[0]] = '"' + feature[1] + '"\n'
            except:
                pass
            finally:
                pass

    for ft in featureDict.items():
        featureFormula.append(' = '.join(ft))
    put_data_in_a_table(path, files)
    if len(featureDict) != (existAttr + 1):
        with open(path + "/attributes.py", "wt", encoding="utf-8-sig", newline="") as attr:
            attr.writelines(sorted(featureFormula))
        put_data_in_a_table(path, files)
    return featureDict


def put_data_in_a_table(path, files):
    attributesMatrix = []
    attributesIndex = dict()
    with open(path + "/attributes.py", encoding="utf-8-sig") as attrFile:
        allAttr = attrFile.readlines()
        emptyRow = ["" for i in range(len(allAttr))]
        firstRow = emptyRow.copy()
        for attrNo in range(len(allAttr)):
            attrCode = allAttr[attrNo].split(" = ")[0]
            attributesIndex[attrCode] = attrNo
            firstRow[attrNo] = attrCode
    attributesMatrix.append(firstRow)
    for file in files:
        currentAttrMap = dict()
        with open(path + '/' + file, encoding="utf-8-sig") as f:
            csvFile = list(csv.reader(f))
            features = csvFile[0]
            for feature in features:
                featureCode = feature.split(' ')[0]
                if featureCode in attributesIndex:
                    currentAttrMap[featureCode] = (features.index(feature), attributesIndex[featureCode])
            for row in range(1, len(csvFile)):
                newRow = emptyRow.copy()
                for column in range(len(csvFile[0])):
                    currentFeature = features[column].split(' ')[0]
                    if currentFeature:
                        attrMap = currentAttrMap[currentFeature]
                        newRow[attrMap[1]] = csvFile[row][column]
                attributesMatrix.append(newRow)
    with open(path + '/putTogether.csv', "w", encoding="utf-8-sig") as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(attributesMatrix)
