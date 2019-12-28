import re
import random
import numpy as np

def textParse(text):
    listOfTokens = re.split(r'\W*', text)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)
def setOfWordVectors(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("单词：%s不在词表中！" % word)
    return returnVec

def trainBayes(trainMatrix, trainCategory):
    #计算先验概率和条件概率
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num, p1Num = np.ones(numWords), np.ones(numWords)
    p0All, p1All = 2.0, 2.0
    for i in range (numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1All += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0All += sum(trainMatrix[i])
    p0_condition = np.log(p0Num / p0All)
    p1_condition = np.log(p1Num / p1All)
    return p0_condition, p1_condition, pAbusive
def spamTest():
    vocab = set()
    docList, fullText, classList = [], [], [] #每个文档对应一个类别
    for i in range(25):
        wordList = textParse(open('email/ham/%s.txt' % str(i+1)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/spam/%s.txt' % str(i+1)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainSet, testSet = list(range(50)), [] #构建trainSet和testSet索引
    for i in range(10):
        randIdx = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIdx])
        del (trainSet[randIdx])
    trainMat, trainClasses = [], []
    for docIndex in trainSet:
        trainMat.append(setOfWordVectors(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainBayes(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWordVectors(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("错误率是：{:.2f}".format(float(errorCount)/len(testSet)))

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #给一句话，预测是不是垃圾——比较p0和p1的概率

    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    print(p1, p0)
    if p1 > p0:
        return 1
    else:
        return 0
#1.处理数据，构造vocab
#2.生成词向量
#3.训练（计算先验概率、条件概率）
#4.测试
if __name__ == '__main__':
    spamTest()