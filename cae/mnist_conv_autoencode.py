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


def get_picture_array(X, index):
    array = X[index].reshape(28,28)
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
    fname = 'mnist/mnist.pkl.gz'
    if not os.path.isfile(fname):
        testfile = urllib.URLopener()
        testfile.retrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", fname)
    f = gzip.open(fname, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    X, y = train_set
    X = np.rint(X * 256).astype(np.int).reshape((-1, 1, 28, 28))  # convert to (0,255) int range (we'll do our own scaling)
    mu, sigma = np.mean(X.flatten()), np.std(X.flatten())

    X_train = X.astype(np.float64)
    X_train = (X_train - mu) / sigma
    X_train = X_train.astype(np.float32)

    # we need our target to be 1 dimensional
    X_out = X_train.reshape((X_train.shape[0], -1))

    conv_filters = 32
    deconv_filters = 32
    filter_size = 7
    epochs = 20
    encode_size = 40
    layerParam= [
        (layers.InputLayer, {'name': 'input_layer', 'shape': (None, 1, 28, 28)}),
        (layers.Conv2DLayer, {'name': 'conv', 'num_filters': conv_filters, 
            'filter_size': (filter_size, filter_size), 'nonlinearity': None}),
        (layers.MaxPool2DLayer, {'name': 'pool', 'pool_size': (2, 2)}),
        (layers.ReshapeLayer, {'name': 'flatten', 'shape': (([0], -1))}),
        (layers.DenseLayer, {'name': 'encode_layer', 'num_units': encode_size}),
        (layers.DenseLayer, {'name': 'hidden', 
            'num_units': deconv_filters * (28 +filter_size - 1)**2 /4}),
        (layers.ReshapeLayer, {'name': 'unflatten', 
            'shape': (([0], deconv_filters, (28 + filter_size - 1) / 2, (28 + filter_size - 1) / 2 ))}),
        (Unpool2DLayer, {'name': 'unpool', 'ds': (2, 2)}),
        (layers.Conv2DLayer, {'name': 'deconv', 'num_filters': 1,
            'filter_size': (filter_size, filter_size), 'nonlinearity': None}),
        (layers.ReshapeLayer, {'name': 'output_layer', 'shape': (([0], -1))})
    ]

    ae = NeuralNet(
        layers=layerParam,
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

    # handle the default limitation of pickle
    sys.setrecursionlimit(10000)
    pickle.dump(ae, open('mnist/conv_ae.pkl','w'))
    # ae = pickle.load(open('mnist/conv_ae.pkl','r'))
    ae.save_params_to('mnist/conv_ae.np')


    X_train_pred = ae.predict(X_train).reshape(-1, 28, 28) * sigma + mu
    X_pred = np.rint(X_train_pred).astype(int)
    X_pred = np.clip(X_pred, a_min = 0, a_max = 255)
    X_pred = X_pred.astype('uint8')
    print X_pred.shape , X.shape


    ###  show random inputs / outputs side by side

    for i in range(0, 10):
        get_random_images(X, X_pred, i)

    return

    ## we find the encode layer from our ae, and use it to define an encoding function

    encode_layer_index = map(lambda pair : pair[0], ae.layers).index('encode_layer')
    print '----------encode_layer_index:', encode_layer_index
    encode_layer = ae.get_all_layers()[encode_layer_index]

    def get_output_from_nn(last_layer, X):
        indices = np.arange(128, X.shape[0], 128)
        sys.stdout.flush()

        # not splitting into batches can cause a memory error
        X_batches = np.split(X, indices)
        out = []
        for count, X_batch in enumerate(X_batches):
            out.append(layers.get_output(last_layer, X_batch).eval())
            sys.stdout.flush()
        return np.vstack(out)

    def encode_input(X):
        return get_output_from_nn(encode_layer, X)
    X_encoded = encode_input(X_train)

    next_layer = ae.get_all_layers()[encode_layer_index + 1]
    final_layer = ae.get_all_layers()[-1]
    new_layer = layers.InputLayer(shape = (None, encode_layer.num_units))

    # N.B after we do this, we won't be able to use the original autoencoder , as the layers are broken up
    next_layer.input_layer = new_layer

    def decode_encoded_input(X):
        return get_output_from_nn(final_layer, X)

    X_decoded = decode_encoded_input(X_encoded) * sigma + mu

    X_decoded = np.rint(X_decoded ).astype(int)
    X_decoded = np.clip(X_decoded, a_min = 0, a_max = 255)
    X_decoded  = X_decoded.astype('uint8')
    print X_decoded.shape

    ### check it worked :

    for i in range(10):
        pic_array = get_picture_array(X_decoded, np.random.randint(len(X_decoded)))
        image = Image.fromarray(pic_array)
        image.save('data/t_' + str(i) + '.png', format="PNG")


    return

if __name__ == '__main__':
    main()




