import sys
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from scipy.cluster import vq
from scipy.misc import imresize
import PCV
from PCV.tools import ncut
from PCV.tools import rof


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


def pKmeans2():
    im = Image.open('../img/3.png')
    im = np.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    m, n = im.shape[0:2]
    wid = 300
    rim = imresize(im, (wid, wid), interp='bilinear')
    rim = np.array(rim, 'f')
    rim = rim.reshape(-1, 3)
    print rim.shape

    rim = vq.whiten(rim)
    centroids, distortion = vq.kmeans(rim, 3)
    code, distance = vq.vq(rim, centroids)
    # code = code.reshape(m, n)
    code = code.reshape(wid, wid)
    code = imresize(code, (m, n), interp='nearest')

    im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)
    plt.figure()
    plt.subplot(211)
    plt.imshow(code)
    plt.gray()
    plt.subplot(212)
    plt.imshow(im)
    plt.gray()
    plt.show()

    return

if __name__ == '__main__':
    # pKmeans()
    pKmeans2()
