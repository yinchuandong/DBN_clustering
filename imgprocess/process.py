import sys
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from skimage import exposure


def testCV():
    # im = cv2.imread('../img/1.png')
    im = Image.open('../img/1.png')
    im = np.array(im)
    print np.shape(im)
    imggray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    print np.shape(imggray)
    ret, thresh = cv2.threshold(imggray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    cv2.polylines(im, cnt, True, (0, 255, 0))
    # print np.shape(cnt)
    # x,y,w,h = cv2.boundingRect(cnt)
    # cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.imshow("Show", im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    plt.imshow(im)
    plt.show()
    return


def testCanny():
    img = Image.open('../img/3.png')
    img = np.array(img)
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, imgbw) = cv2.threshold(imggray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edges = cv2.Canny(imgbw, 100, 200)
    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.axis('off')
    plt.subplot(222), plt.imshow(imggray, cmap='gray')
    plt.title('gray Image'), plt.axis('off')
    plt.subplot(223), plt.imshow(imgbw, cmap='gray')
    plt.title('binary Image'), plt.axis('off')
    plt.subplot(224), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.axis('off')

    plt.show()
    return


def testSkimage():
    img = Image.open('../img/1.png')
    img = np.array(img)
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # (thresh, imgbw) = cv2.threshold(imggray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # canny detector
    # from skimage.feature import canny
    # edges = canny(imggray/ 255.)
    from scipy import ndimage as ndi
    # fill_imgbw = ndi.binary_fill_holes(edges)
    # label_objects, nb_labels = ndi.label(fill_imgbw)
    # sizes = np.bincount(label_objects.ravel())
    # mask_sizes = sizes > 20
    # mask_sizes[0] = 0
    # cleaned_imgbw = mask_sizes[label_objects]

    markers = np.zeros_like(imggray)
    markers[imggray < 120] = 1
    markers[imggray > 150] = 2

    from skimage.filters import sobel
    elevation_map = sobel(imggray)
    from skimage.morphology import watershed
    segmentation = watershed(elevation_map, markers)

    # from skimage.color import label2rgb
    # segmentation = ndi.binary_fill_holes(segmentation - 10)
    # labeled_coins, _ = ndi.label(segmentation)
    # image_label_overlay = label2rgb(labeled_coins, image=imggray)
    plt.imshow(segmentation, cmap='gray')
    plt.show()
    return


def testSkimage2():
    img = Image.open('../img/1.png')
    # img = Image.open('../img/2.png')
    img = np.array(img)
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # (thresh, imgbw) = cv2.threshold(imggray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    camera = imggray
    # camera = data.camera()
    val = filters.threshold_otsu(camera)

    hist, bins_center = exposure.histogram(camera)

    plt.figure(figsize=(9, 4))
    plt.subplot(131)
    plt.imshow(camera, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(camera < val, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.subplot(133)
    plt.plot(bins_center, hist, lw=2)
    plt.axvline(val, color='k', ls='--')

    plt.tight_layout()
    plt.show()
    return


def testSIFT():
    img = Image.open('../img/4.png')
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(img, None)
    img = cv2.drawKeypoints(img, kp)
    plt.figure() 
    plt.imshow(img)
    plt.show()

    return


if __name__ == '__main__':
    print 'start'
    # test()
    # testCV()
    testCanny()
    # testSkimage()
    # testSkimage2()
    # testSIFT()
    print 'end'
