from dataPretreatment import normalized
import csv

readPwd = 'D:/Documents/drillData/readyToUse/'
writePwd = 'D:/Documents/Programing/Github/Python/machineLearning/dataSet'

readFileName = 'drillData_v3_before.csv'
writeFileName = 'drillData_v3.csv'
handle = normalized.HandleUsefulData(readPwd + readFileName, readPwd + writeFileName).v2()

