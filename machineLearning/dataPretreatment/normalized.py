import numpy as np
import os
import csv


# 数据归一化方法：
# 1.效益型 越大越好-----------	> 大的大
# 2.成本型 越小越好-----------	> 小的大
# 3.固定型 越接近某个值越好	---->
# 4.偏离型 约远离某个值越好----->
# 5.区间型 越靠近某个区间越好--->

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
        with open(path + "/attributes.py", "wt", encoding="utf-8-sig") as attr:
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
    with open(path + '/putTogether.csv', "w") as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(attributesMatrix)


