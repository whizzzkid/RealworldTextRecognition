import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K

from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os import listdir
import os.path
import numpy as np
import string

now = datetime.datetime.now

data_dir = "data/Cropped_Words/COCO-Text-words-trainval"
training_dir = "/train_words/"
valid_dir = "/val_words/"

def generate_vocab():    
    f = open(data_dir + '/train_words_gt.txt', 'r')
    labels = []
    for line in f.readlines():
        line = line.replace('|','')
        lab = line.split(',')
        if len(lab) >=2:            
            labels.append(lab[1])
            
    f = open(data_dir + '/val_words_gt.txt', 'r')    
    for line in f.readlines():
        line = line.replace('|','')
        lab = line.split(',')
        if len(lab) >=2:            
            labels.append(lab[1])
            
    vocab = list(set(labels))
    print("Unique vocabulary in the training and validation set is: {}".format(len(vocab)))
    return vocab

def training_generator(batch_size, num_classes):
    while True:
        idx = np.random.randint(len(train_data), size = batch_size)
        x = train_data[idx,:]
        y = train_label[idx]
        x_train = x.astype('float32')
        x_train /= 255
        y_train = keras.utils.to_categorical(y, num_classes)        
        yield (x_train, y_train)
        
                                

def train_model(model, train, test, num_classes):
    x_train = train[0]
    x_test = test[0]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    t = now()
    model.fit(x_train, y_train,
              batch_size=1000,
              epochs=4,
              verbose=1,
              validation_data=(x_test, y_test))
    print('Training time: %s' % (now() - t))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


npzfile = np.load('words.npz')
print (npzfile.files)
train_data = npzfile['train_data']
train_label= npzfile['train_label']
valid_data= npzfile['valid_data']
valid_label = npzfile['valid_label']
vocab = generate_vocab()

network_layers = [
    ZeroPadding2D((1,1), input_shape=(32,100,1)),
    Conv2D(64, 5, activation='relu'),    
    MaxPooling2D(pool_size=(2,2)),
    ZeroPadding2D((1,1)),
    Conv2D(128, 5, activation='relu'),    
    MaxPooling2D(pool_size=(2,2), dim_ordering="th"),
    ZeroPadding2D((1,1)),
    Conv2D(256, 3, activation='relu'),    
    MaxPooling2D(pool_size=(2,2), dim_ordering="th"),
    ZeroPadding2D((1,1)),
    Conv2D(512, 3, activation='relu'),
    ZeroPadding2D((1,1)),
    Conv2D(512, 3, activation='relu'),    
    Flatten(),
    Dense(4096, activation='relu'),   
    Dense(4096, activation='relu'),  
    Dense(88172, activation ='softmax')
]

# network_layers = [
#     ZeroPadding2D((1,1), input_shape=(32,100,1)),
#     Conv2D(64, 5, activation='relu'),    
#     Flatten(),
#     Dense(100, input_shape=(32,100,1), activation='relu'),  
#     Dense(len(vocab), activation ='softmax')
# ]

# create complete model
model = Sequential(network_layers)

x_test = valid_data.astype('float32')
x_test /=255
y_test = keras.utils.to_categorical(valid_label, len(vocab))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.summary()
t = now()
model.fit_generator(training_generator(100,len(vocab)), steps_per_epoch=1,epochs=10000)