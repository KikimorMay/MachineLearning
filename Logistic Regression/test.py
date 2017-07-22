import logRegres
import numpy as np
dataArr, labelArr = logRegres.loadDataSet()
weight = logRegres.stocDradAscent0(np.array(dataArr), labelArr)
logRegres.plotBestFit(weight)
