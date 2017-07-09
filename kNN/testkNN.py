import kNN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from os import listdir


    #  对原始数据画图
# fig = plt.figure()
# ax = fig.add_subplot(111)       #参数111的意思是：将画布分割成1行1列，图像画在从左到右从上到下的第1块
# ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*np.array(datingLabels),  15.0*np.array(datingLabels))
# plt.show()

# kNN.datingClassTest()

  # test img2vector()
# testVector = kNN.img2vector('trainingDigits/0_0.txt')
# print(testVector)

trainingFileList = listdir('trainingDigits')  # 从文件夹中获得目录列表
fileStr = trainingFileList[0]
print(open('trainingDigits/%s' %fileStr))

kNN.handwritingClasstest()

