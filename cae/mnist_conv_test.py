# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import theano.tensor as T
from nolearn.lasagne import BatchIterator

import pickle
import sys
import os
import urllib
import gzip
import cPickle
from PIL import Image


# <codecell>

class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(2, axis=2).repeat(2, axis=3)


# <codecell>

### when we load the batches to input to the neural network, we randomly / flip
### rotate the images, to artificially increase the size of the training set

class FlipBatchIterator(BatchIterator):

    def transform(self, X1, X2):
        X1b, X2b = super(FlipBatchIterator, self).transform(X1, X2)
        X2b = X2b.reshape(X1b.shape)

        bs = X1b.shape[0]
        h_indices = np.random.choice(bs, bs / 2, replace=False)  # horizontal flip
        v_indices = np.random.choice(bs, bs / 2, replace=False)  # vertical flip

        ###  uncomment these lines if you want to include rotations (images must be square)  ###
        #r_indices = np.random.choice(bs, bs / 2, replace=False) # 90 degree rotation
        for X in (X1b, X2b):
            X[h_indices] = X[h_indices, :, :, ::-1]
            X[v_indices] = X[v_indices, :, ::-1, :]
            #X[r_indices] = np.swapaxes(X[r_indices, :, :, :], 2, 3)
        shape = X2b.shape
        X2b = X2b.reshape((shape[0], -1))

        return X1b, X2b

# <codecell>

def get_picture_array(X, index):
    array = X[index].reshape(256,256)
    array = np.clip(array, a_min = 0, a_max = 255)
    return  array.repeat(4, axis = 0).repeat(4, axis = 1).astype(np.uint8())

def get_random_images(X, X_pred, i=0):
    index = np.random.randint(5000)
    print index
    original_image = Image.fromarray(get_picture_array(X, index))
    new_size = (original_image.size[0] * 2, original_image.size[1])
    new_im = Image.new('L', new_size)
    new_im.paste(original_image, (0,0))
    rec_image = Image.fromarray(get_picture_array(X_pred, index))
    new_im.paste(rec_image, (original_image.size[0],0))
    new_im.save('data/test_' + str(i) + '.png', format="PNG")



def main():
    # load data set
    fname = 'imagenet.pkl'
    train_set = pickle.load(open(fname, 'r'))
    X = train_set[0:1000]
    # X = X.astype(np.int).reshape((-1, 3, 256, 256))  # convert to (0,255) int range (we'll do our own scaling)
    # sigma = np.std(X.flatten())
    # mu = np.mean(X.flatten())

    # print np.shape(X[0])
    # print mu
    # print sigma
    # return
    # # <codecell>
    # X_train = X.astype(np.float64)
    # X_train = (X_train - mu) / sigma
    # X_train = X_train.astype(np.float32)
    X_train = X.astype(np.float32)

    # we need our target to be 1 dimensional
    X_out = X_train.reshape((X_train.shape[0], -1))

    # <codecell>
    conv_filters = 32
    deconv_filters = 32
    filter_sizes = 7
    epochs = 20
    encode_size = 40
    ae = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv', layers.Conv2DLayer),
            ('pool', layers.MaxPool2DLayer),
            ('flatten', layers.ReshapeLayer),  # output_dense
            ('encode_layer', layers.DenseLayer),
            ('hidden', layers.DenseLayer),  # output_dense
            ('unflatten', layers.ReshapeLayer),
            ('unpool', Unpool2DLayer),
            ('deconv', layers.Conv2DLayer),
            ('output_layer', layers.ReshapeLayer),
            ],
        input_shape=(None, 3, 256, 256),
        conv_num_filters=conv_filters, conv_filter_size = (filter_sizes, filter_sizes),
        conv_nonlinearity=None,
        pool_pool_size=(2, 2),
        flatten_shape=(([0], -1)), # not sure if necessary?
        encode_layer_num_units = encode_size,
        hidden_num_units= deconv_filters * (256 + filter_sizes - 1) ** 2 / 4,
        unflatten_shape=(([0], deconv_filters, (256 + filter_sizes - 1) / 2, (256 + filter_sizes - 1) / 2 )),
        unpool_ds=(2, 2),
        deconv_num_filters=1, deconv_filter_size = (filter_sizes, filter_sizes),
        deconv_nonlinearity=None,
        output_layer_shape = (([0], -1)),
        update_learning_rate = 0.01,
        update_momentum = 0.975,
        batch_iterator_train=FlipBatchIterator(batch_size=128),
        regression=True,
        max_epochs= epochs,
        verbose=1,
        )
    ae.fit(X_train, X_out)
    print '---------------train end'
    print
    ###  expect training / val error of about 0.087 with these parameters
    ###  if your GPU not fast enough, reduce the number of filters in the conv/deconv step

    # <codecell>


    pickle.dump(ae, open('mnist/my_conv_ae.pkl','w'))
    # ae = pickle.load(open('mnist/my_conv_ae.pkl','r'))
    ae.save_params_to('mnist/my_conv_ae.np')

    # <codecell>

    X_train_pred = ae.predict(X_train).reshape(-1, 256, 256) * sigma + mu
    X_pred = np.rint(X_train_pred).astype(int)
    X_pred = np.clip(X_pred, a_min = 0, a_max = 255)
    X_pred = X_pred.astype('uint8')
    print X_pred.shape , X.shape

    # <codecell>

    ###  show random inputs / outputs side by side

    for i in range(0, 10):
        get_random_images(X, X_pred, i)

    return


    return


def genDataset():
    base_path = '../data_exterior/exterior/'
    files = os.listdir(base_path)
    imgArr = []
    for filename in files:
        img = Image.open(base_path + filename)
        # arr = np.array(img.getdata()).flatten()
        arr = np.array(img)
        m = np.shape(arr)
        if len(m) != 3 or m[2] != 3:
            print 'invalid file: ', filename
            continue
        # print np.shape(arr)
        arr = np.swapaxes(arr, 0, 2)
        arr = np.swapaxes(arr, 1, 2)
        # print np.shape(arr)
        imgArr.append(arr)
        # break
    imgArr = np.array(imgArr)
    print np.shape(imgArr)
    fh = open('imagenet.pkl', 'w')
    pickle.dump(imgArr, fh)
    fh.close()
    return




if __name__ == '__main__':
    main()
    # genDataset()



