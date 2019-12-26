'''
Created on Jun 14, 2011

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    with open(fileName) as f:
        stringArr = [line.strip().split(delim) for line in f.readlines()]
        datArr = [list(map(float, line)) for line in stringArr]
        return mat(datArr)
    
def pca(dataMat, topNfeat=9999999):
    #1.去平均值
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    #2.从大到小对N个值进行排序
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    #3.将数据转换到新空间
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals #重构数据
    return lowDDataMat, reconMat
def replaceNanWithMean():
    dataMat = loadDataSet('secom.data', ' ')
    numFeat = shape(dataMat)[1] #计算特征数
    for i in range(numFeat):
        #1.计算所有特征的平均值
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0], i])
        #2.将所有Nan置为平均值
        dataMat[nonzero(isnan(dataMat[:,i].A))[0], i] = meanVal
    return dataMat
    
dataMat = replaceNanWithMean()

#below is a quick hack copied from pca.pca()
meanVals = mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals #remove mean
covMat = cov(meanRemoved, rowvar=0)
eigVals,eigVects = linalg.eig(mat(covMat))
eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
eigValInd = eigValInd[::-1]#reverse
sortedEigVals = eigVals[eigValInd]
total = sum(sortedEigVals)
varPercentage = sortedEigVals/total*100

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1, 21), varPercentage[:20], marker='^')
plt.xlabel('Principal Component Number')
plt.ylabel('Percentage of Variance')
plt.show()