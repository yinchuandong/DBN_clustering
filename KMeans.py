import numpy as np


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
    def __init__(self, dataMat):
        self.dataMat = dataMat
        return

    def cluster(self, k, distMeas=distEclud, createCenter=randCenter):
        m, n = np.shape(self.dataMat)
        labelMat = np.mat(np.zeros((m, 2)))  # cluster indicator
        centerMat = createCenter(self.dataMat, k)
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):
                minDist = np.inf
                minIndex = -1
                for j in range(k):
                    distIJ = distEclud(self.dataMat[i], centerMat[j])
                    if distIJ < minDist:
                        minDist = distIJ
                        minIndex = j
                if labelMat[i, 0] != minIndex:
                    clusterChanged = True
                labelMat[i, :] = minIndex, minDist**2

            for cent in range(k):
                # get all points in the same cluster
                ptsInCluster = self.dataMat[np.nonzero(labelMat[:, 0].A == cent)[0]]
                centerMat[cent, :] = np.mean(ptsInCluster, axis=0)
        return centerMat, labelMat

    def cost(self, centerMat, labelMat):
        label = np.asarray(labelMat[:, 0].astype(int).T)[0]
        # print label
        # print centerMat[label]
         
        return


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
    km = KMeans(dataMat)
    centerMat, labelMat = km.cluster(3)
    km.cost(centerMat, labelMat)






