'''This example uses a convolutional stack followed by a recurrent stack
and a CTC logloss function to perform optical character recognition
of generated text images. I have no evidence of whether it actually
learns general shapes of text, or just is able to recognize all
the different fonts thrown at it...the purpose is more to demonstrate CTC
inside of Keras.  Note that the font list may need to be updated
for the particular OS in use.

This starts off with 4 letter words.  For the first 12 epochs, the
difficulty is gradually increased using the TextImageGenerator class
which is both a generator class for test/train data and a Keras
callback class. After 20 epochs, longer sequences are thrown at it
by recompiling the model to handle a wider image and rebuilding
the word list to include two words separated by a space.

The table below shows normalized edit distance values. Theano uses
a slightly different CTC implementation, hence the different results.

            Norm. ED
Epoch |   TF   |   TH
------------------------
    10   0.027   0.064
    15   0.038   0.035
    20   0.043   0.045
    25   0.014   0.019

This requires cairo and editdistance packages:
pip install cairocffi
pip install editdistance

Created by Mike Henry
https://github.com/mbhenry/
'''
import os
import itertools
import re
import datetime
import numpy as np
from scipy import ndimage
import pylab
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks

from mjsynth_dictnet import MJSYNTH_DICTNET
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        # 26 is space, 27 is CTC blank char
        outstr = ''
        for c in out_best:
            if c >= 0 and c < 26:
                outstr += chr(c + ord('a'))
            elif c == 26:
                outstr += ' '
        ret.append(outstr)
    return ret

def train(run_name, start_epoch, stop_epoch, img_w):
    # Input Parameters
    img_h = 32
    words_per_epoch = 16000
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_h, img_w)
    else:
        input_shape = (img_h, img_w, 1)   

   
    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(26, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    # labels = Input(name='the_labels', shape=[16], dtype='float32')
    # input_length = Input(name='input_length', shape=[1], dtype='int64')
    # label_length = Input(name='label_length', shape=[1], dtype='int64')
    # # Keras doesn't currently support loss funcs with extra parameters
    # # so CTC loss is implemented in a lambda layer
    # loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # # clipnorm seems to speeds up convergence
    # sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    # model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    # if start_epoch > 0:
    #     weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
    #     model.load_weights(weight_file)
    # # captures output of softmax so we can decode the output during visualization
    # test_func = K.function([input_data], [y_pred])

    # viz_cb = VizCallback(run_name, test_func, img_gen.next_val())

    # model.fit_generator(generator=img_gen.next_train(), steps_per_epoch=(words_per_epoch - val_words),
    #                     epochs=stop_epoch, validation_data=img_gen.next_val(), validation_steps=val_words,
    #                     callbacks=[viz_cb, img_gen], initial_epoch=start_epoch)


def text_to_labels(text, num_classes):
    ret = []
    for char in text:
        print (char)
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
        else:
            print(char, " not found")

    return ret


if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    #train(run_name, 0, 20, 100)
    # increase to wider images and start at epoch 20. The learned weights are reloaded
    #train(run_name, 20, 25, 512)
    
    lexicon = np.genfromtxt('../data/mnt/ramdisk/max/90kDICT32px/lexicon.txt', dtype='str' )
    print ("Lexicon size: {}".format(lexicon.shape))
    print ("Largest string in lexicon: {}".format(len(max(lexicon, key=len))))

    z = MJSYNTH_DICTNET("train",2,10,[])
    for i,y in enumerate(z.labels):
        print (i,y,z.classes[y[0]])
    print (z.class_mapping)
    print (z.x.shape)

    gs = gridspec.GridSpec(min(5,int(z.x.shape[0]/4)),4, top=1., bottom=0., right=1., left=0., hspace=0.1, wspace=0.1)
    for i,g in enumerate(gs): 
        ax = plt.subplot(g)
        ax.imshow(z.x[i,:,:,0], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.set_title("Class: " + lexicon[int(z.classes[z.labels[i][0]])])
    plt.show()
    