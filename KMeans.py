import numpy as np
import theano
import theano.tensor as T

import sys
import warnings
warnings.simplefilter("error")


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCenter(dataMat, k):
    n = np.shape(dataMat)[1]
    centerMat = np.mat(np.zeros((k, n)))
    for j in range(n):
        min = np.min(dataMat[:, j])
        rangeJ = float(np.max(dataMat[:, j]) - min)
        centerMat[:, j] = min + rangeJ * np.random.rand(k, 1)
    return centerMat


class KMeans(object):
    def __init__(self, X, C):
        self.X = X
        self.C = C
        return

    def cluster(self, dataMat, k, distMeas=distEclud, createCenter=randCenter):
        m, n = np.shape(dataMat)
        labelMat = np.mat(np.zeros((m, 2)))  # cluster indicator
        centerMat = createCenter(dataMat, k)
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):
                minDist = np.inf
                minIndex = -1
                for j in range(k):
                    distIJ = distEclud(dataMat[i], centerMat[j])
                    if distIJ < minDist:
                        minDist = distIJ
                        minIndex = j
                if labelMat[i, 0] != minIndex:
                    clusterChanged = True
                labelMat[i, :] = minIndex, minDist**2

            for cent in range(k):
                # get all points in the same cluster
                ptsInCluster = dataMat[np.nonzero(labelMat[:, 0].A == cent)[0]]
                centerMat[cent, :] = np.mean(ptsInCluster, axis=0)
        return centerMat, labelMat

    def bisCluster(self, dataMat, k, distMeas=distEclud):
        """
        Bisecting k-means
        """
        m = np.shape(dataMat)[0]
        labelMat = np.mat(np.zeros((m, 2)))
        centroid0 = np.mean(dataMat, axis=0).tolist()[0]
        centList = [centroid0]  # create a list with one centroid
        for j in range(m):  # calc initial Error
            labelMat[j, 1] = distMeas(np.mat(centroid0), dataMat[j, :])**2
        while (len(centList) < k):
            lowestSSE = np.inf
            for i in range(len(centList)):
                # get the data points currently in cluster i
                ptsInCurrCluster = dataMat[np.nonzero(labelMat[:, 0].A == i)[0], :]
                centroidMat, splitClustAss = self.cluster(ptsInCurrCluster, 2, distMeas)
                sseSplit = np.sum(splitClustAss[:, 1])  # compare the SSE to the currrent minimum
                sseNotSplit = np.sum(labelMat[np.nonzero(labelMat[:, 0].A != i)[0], 1])
                # print "sseSplit, and notSplit: ", sseSplit, sseNotSplit
                if (sseSplit + sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitClustAss.copy()
                    lowestSSE = sseSplit + sseNotSplit
            bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # change 1 to 3,4, or whatever
            bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
            # print 'the bestCentToSplit is: ', bestCentToSplit
            # print 'the len of bestClustAss is: ', len(bestClustAss)
            centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
            centList.append(bestNewCents[1, :].tolist()[0])
            # reassign new clusters, and SSE
            labelMat[np.nonzero(labelMat[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
        return np.mat(centList), labelMat

    def cost(self):
        R = T.sqrt(T.sum(T.pow(self.X - self.C, 2)))
        return R


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

if __name__ == '__main__':
    dataMat = loadDataSet('testSet2.txt')
    dataMat = np.mat(dataMat)

    X = T.matrix('X')
    C = T.matrix('C')
    index = T.lscalar('index')

    km = KMeans(X, C)
    centerMat, labelMat = km.bisCluster(dataMat, 3)

    

    label = np.asarray(labelMat[:, 0].astype(int).T)[0]
    CMat = theano.shared(np.asarray(centerMat[label], dtype=theano.config.floatX
                                    ), borrow=True)
    XMat = theano.shared(np.asarray(dataMat, dtype=theano.config.floatX), borrow=True)
    cost = km.cost()
    fn = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            X: XMat[index:4],
            C: CMat[index:4]
        }
    )

    result = fn(0)
    print result

    X = dataMat[0: 4]
    C = centerMat[label][0:4]
    r2 = X - C
    print r2
    cont = 0
    for i in range(1000):
        r = np.sqrt(np.sum(np.power(r2, 2)))
        if r > 2:
            cont = cont + 1
    print cont
