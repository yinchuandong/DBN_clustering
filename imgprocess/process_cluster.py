import sys
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from scipy.cluster import vq
from scipy.misc import imresize
import PCV
from PCV.tools import ncut, rof, pca


def pKmeans(k=2):
    im = Image.open('../img/3.png')
    im = np.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    m, n = im.shape[0:2]
    wid = 50
    rim = imresize(im, (wid, wid), interp='bilinear')
    rim = np.array(rim, 'f')

    A = ncut.ncut_graph_matrix(rim, sigma_d=1, sigma_g=1e-2)
    # print A.shape
    code, V = ncut.cluster(A, k=3, ndim=3)
    codeim = imresize(code.reshape(wid, wid), (m, n), interp='nearest')
    # print codeim.shape
    plt.figure()
    plt.subplot(211)
    plt.imshow(codeim)
    plt.gray()
    plt.subplot(212)
    plt.imshow(im)
    plt.gray()
    plt.show()
    return


def pKmeans2(filename, k=3):
    im = Image.open(filename)
    im = np.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    m, n = im.shape[0:2]
    wid = 150
    rim = imresize(im, (wid, wid), interp='bilinear')
    rim = np.array(rim, 'f')
    rim = rim.reshape(-1, 3)
    # print rim.shape

    rim = vq.whiten(rim)
    centroids, distortion = vq.kmeans(rim, k)
    code, distance = vq.vq(rim, centroids)
    code = code.reshape(wid, wid)
    code = imresize(code, (m, n), interp='nearest')
    labels = np.unique(code)
    im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)

    # split one image to k images according to cluster
    arr = np.full((k, m, n, 3), 255, dtype=np.uint8)
    for i in range(k):
        pts = np.where(code[:][:] == labels[i])
        for j in range(len(pts[0])):
            row = pts[0][j]
            col = pts[1][j]
            arr[i][row][col] = im[row][col]

    gray0 = cv2.cvtColor(arr[0], cv2.COLOR_RGB2GRAY)
    gray0 = sizeFilter(gray0)
    # gray0 = cv2.medianBlur(gray0, 3)
    # gray0 = cv2.GaussianBlur(gray0, (5, 5), 1.5)
    edge0 = cv2.Canny(gray0, 100, 200)
    gray1 = cv2.cvtColor(arr[1], cv2.COLOR_RGB2GRAY)
    gray1 = sizeFilter(gray1)
    # gray1 = cv2.medianBlur(gray1, 3)
    # gray1 = cv2.GaussianBlur(gray1, (5, 5), 1.5)
    edge1 = cv2.Canny(gray1, 100, 200)
    gray2 = cv2.cvtColor(arr[2], cv2.COLOR_RGB2GRAY)
    gray2 = sizeFilter(gray2)
    # gray2 = cv2.medianBlur(gray2, 3)
    # gray2 = cv2.GaussianBlur(gray2, (5, 5), 1.5)
    edge2 = cv2.Canny(gray2, 100, 200)

    sift = cv2.SIFT()
    kp0, des0 = sift.detectAndCompute(gray0, None)
    arr[0] = cv2.drawKeypoints(arr[0], kp0)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    arr[1] = cv2.drawKeypoints(arr[1], kp1)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    arr[2] = cv2.drawKeypoints(arr[2], kp2)

    print 'sift: ', len(kp0), len(kp1), len(kp2)
    print 'edges: '
    print len(np.where(edge0.flatten() == 255)[0])
    print len(np.where(edge1.flatten() == 255)[0])
    print len(np.where(edge2.flatten() == 255)[0])

    # for splited image
    plt.figure(figsize=(10, 10))
    plt.subplot(321), plt.axis('off'), plt.gray()
    plt.imshow(arr[0]), plt.title('cluster 0')
    plt.subplot(322), plt.axis('off'), plt.gray()
    plt.imshow(edge0), plt.title('edge 0')
    plt.subplot(323), plt.axis('off'), plt.gray()
    plt.imshow(arr[1]), plt.title('cluster 1')
    plt.subplot(324), plt.axis('off'), plt.gray()
    plt.imshow(edge1), plt.title('edge 1')
    plt.subplot(325), plt.axis('off'), plt.gray()
    plt.imshow(arr[2]), plt.title('cluster 2')
    plt.subplot(326), plt.axis('off'), plt.gray()
    plt.imshow(edge2), plt.title('edge 2')
    plt.show()

    # Image.fromarray(gray0).save('gray0.png')
    # for original image
    # plt.figure()
    # plt.subplot(211), plt.imshow(code), plt.gray()
    # plt.subplot(212), plt.imshow(im), plt.gray()
    # plt.show()
    return


def sizeFilter(inarr, threshold=50):
    outarr = inarr.copy()
    m, n = inarr.shape
    mark = np.zeros_like(inarr)
    stack = []
    for i in range(m):
        for j in range(n):
            if inarr[i, j] == 255 or mark[i, j] == 1:
                continue
            stack.append((i, j, inarr[i, j]))
            area = []
            area.append((i, j))
            while stack:
                pt = stack.pop()
                mark[pt[0], pt[1]] = 1
                if len(area) > threshold:
                    stack = []
                    area = []
                    break
                # print pt
                startY = 0 if pt[0] - 1 < 0 else pt[0] - 1
                startX = 0 if pt[1] - 1 < 0 else pt[1] - 1
                endY = m - 1 if pt[0] + 1 > m - 1 else pt[0] + 1
                endX = n - 1 if pt[1] + 1 > n - 1 else pt[1] + 1
                for ti in range(startY, endY + 1):
                    for tj in range(startX, endX + 1):
                        if inarr[ti, tj] == 255 or mark[ti, tj] == 1:
                            continue
                        stack.append((ti, tj, inarr[ti, tj]))
                        area.append((ti, tj))
                        mark[ti, tj] = 1
            if len(area) == 0:
                continue
            for t in area:
                outarr[t] = 255
            # print 'len: ', len(area)
            # break
        # break
    # plt.figure()
    # plt.subplot(211), plt.imshow(inarr), plt.gray()
    # plt.subplot(212), plt.imshow(outarr), plt.gray()
    # plt.show()
    return outarr

if __name__ == '__main__':
    # pKmeans()
    pKmeans2('../img/2.png')
    # img = np.array(Image.open('gray0.png'))
    # sizeFilter(img, threshold=50)
    


