# 数据归一化方法：
# 1.效益型 越大越好-----------	> 大的大
# 2.成本型 越小越好-----------	> 小的大
# 3.固定型 越接近某个值越好	---->
# 4.偏离型 约远离某个值越好----->
# 5.区间型 越靠近某个区间越好---	>

import numpy as np
#传入numpy.array
#传入每个属性对应的归一化方式 一维向量
def normalize(arr,method,standard=None):
	colomnCount = len(arr[0])
	rowCount = len(arr)

	if (standard != None) and (len(standard) != colomnCount):
		return "error: wrong standard length"

	for colomn in range(colomnCount):
		currentColomn = arr[:,colomn]
		maxOfColomn = np.max(currentColomn)
		minOfColomn = np.min(currentColomn)

		for row in range(rowCount):
			if method[colomn] == 1:
				arr[row,colomn] = (arr[row,colomn] - minOfColomn)/(maxOfColomn - minOfColomn)
			elif method[colomn] == 2:
				arr[row,colomn] = (maxOfColomn - arr[row,colomn])/(maxOfColomn - minOfColomn)
			elif method[colomn] == 3 and standard:
				arr[row,colomn] = (abs(standard[colomn] - arr[row,colomn]))/(minOfColomn - maxOfColomn) + 1
			elif method[colomn] == 4 and standard:
				arr[row,colomn] = (abs(arr[row,colomn]-standard[colomn])/(maxOfColomn - minOfColomn))
			elif method[colomn] == 5:
				pass
			else:
				return "error: wrong normalize method input"
	return arr
