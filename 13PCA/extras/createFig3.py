'''
Created on Jun 1, 2011

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

n = 1000 #number of points to create
xcord0 = []; ycord0 = []
xcord1 = []; ycord1 = []
xcord2 = []; ycord2 = []
markers =[]
colors =[]
fw = open('testSet3.txt','w')
for i in range(n):
    groupNum = int(3*random.uniform())
    [r0,r1] = random.standard_normal(2)
    if groupNum == 0:
        x = r0 + 16.0
        y = 1.0*r1 + x
        xcord0.append(x)
        ycord0.append(y)
    elif groupNum == 1:
        x = r0 + 8.0
        y = 1.0*r1 + x
        xcord1.append(x)
        ycord1.append(y)
    elif groupNum == 2:
        x = r0 + 0.0
        y = 1.0*r1 + x
        xcord2.append(x)
        ycord2.append(y)
    fw.write("%f\t%f\t%d\n" % (x, y, groupNum))

fw.close()
fig = plt.figure()
ax = fig.add_subplot(211)
ax.scatter(xcord0,ycord0, marker='^', s=90)
ax.scatter(xcord1,ycord1, marker='o', s=50,  c='red')
ax.scatter(xcord2,ycord2, marker='v', s=50,  c='yellow')
ax = fig.add_subplot(212)
myDat = loadDataSet('testSet3.txt')
lowDDat,reconDat = pca(myDat[:,0:2],1)
label0Mat = lowDDat[nonzero(myDat[:,2]==0)[0],:2][0] #get the items with label 0
label1Mat = lowDDat[nonzero(myDat[:,2]==1)[0],:2][0] #get the items with label 1
label2Mat = lowDDat[nonzero(myDat[:,2]==2)[0],:2][0] #get the items with label 2
#ax.scatter(label0Mat[:,0],label0Mat[:,1], marker='^', s=90)
#ax.scatter(label1Mat[:,0],label1Mat[:,1], marker='o', s=50,  c='red')
#ax.scatter(label2Mat[:,0],label2Mat[:,1], marker='v', s=50,  c='yellow')
ax.scatter(label0Mat[:,0].tolist(),zeros(shape(label0Mat)[0]), marker='^', s=90)
ax.scatter(label1Mat[:,0].tolist(),zeros(shape(label1Mat)[0]), marker='o', s=50,  c='red')
ax.scatter(label2Mat[:,0].tolist(),zeros(shape(label2Mat)[0]), marker='v', s=50,  c='yellow')
plt.show()