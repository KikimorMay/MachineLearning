import numpy as np


def loadDataSet():
    dataMat = []
    labelMat = []
    for line in open('testSet.txt').readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

def gradAscent(dataMat, labelMat):
    dataMatrix = np.array(dataMat)
    labelMatrix = np.array(labelMat).reshape(100,1)    #得到的是一个列向量
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weight = np.ones((n, 1))                      #所有权重为n行1列
    for k in range(maxCycles):
        h = sigmoid(dataMatrix.dot(weight))          #矩阵相乘，得到一个m行1列的向量
        error = labelMatrix - h
        weight = weight + alpha * dataMatrix.transpose().dot(error)
    return weight

def plotBestFit(weight):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    m = np.shape(dataMat)[0]
    dataArr = np.array(dataMat)
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(m):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weight[0] - weight[1]*x)/weight[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


    #随机梯度上升算法，每次选择一个数据点来更新权重
def stocDradAscent0(dataMatrix, classlabels):
    m, n = np.shape(dataMatrix)
    print(m,n)
    alpha = 0.01
    weight = np.ones(n)
    for i in range(m):
        h = sigmoid(dataMatrix[i].dot(weight.reshape(n, 1)))
        error = classlabels[i] - h
        weight = weight + alpha * error * dataMatrix[i]
    return weight


