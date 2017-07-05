import numpy as np
import operator
from os import listdir

str2num = {
    'largeDoses':3,
    'smallDoses':2,
    'didntLike':1
}

    #  文件转化为numpy数组

def file2matrix(filename):                              #将文本记录转化成Numpy，文本记录的格式已知，此文本每行数据用tab隔开
    fr = open(filename)
    arrayOLines = fr.readlines()                        # readlines()自动将文件内容分析成一个行的列表，该列表可以由 python 的 for... in ... 结构进行处理
    numberOfLines = len(arrayOLines)

    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:                            #循环每一行
        line = line.strip()                             #去掉回车
        listFromLine = line.split('\t')                 #用制表符分开
        returnMat[index,:] = listFromLine[0:3]          #前三个是特征信息
        classLabelVector.append( str2num[listFromLine[-1]] )#最后一个是类别
        index = index + 1
    return returnMat, classLabelVector


    #归一化数据
def autoNorm(dataset):
    minVals = dataset.min(0)            #将每列最小值放在minVals中，minVals是一个向量
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros( np.shape(dataset) )
    normDataSet = dataset - minVals
    normDataSet = normDataSet/ranges
    return normDataSet, ranges, minVals

# def createDataset():
#     group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
#     labels = ['A', 'A', 'B', 'B']
#     return group, labels
######


    #  kNN分类

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]                      #样本个数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  #将需要分类数据变为(dataSetSize,1)，减去训练数据
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)               #每一行加起来,返回的是一个列向量，当axis = 0时，返回的是一个行向量
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()            #排序，将distances中的元素从小到大排列，提取其对应的index
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]      #临近k个样本的类别
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1    #key值对应数字加1  若不存在返回0
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True) #数量最多的类 ,得到的结果是list，list中每个元素是一个tuple
    return sortedClassCount[0][0]


    #测试分类器的效果
def datingClassTest():
    hoRation = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]                                #行数
    numTestVecs = int( m*hoRation )                     #选择10%的数据当做测试数据
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        if(classifierResult != datingLabels[i]):
            errorCount = errorCount + 1
    print('error rate is %f' % (errorCount/numTestVecs))

    #将每一张图像转化为测试向量
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        returnVect[0, i*32:(i+1)*32] = int(lineStr)
    return returnVect

    #
