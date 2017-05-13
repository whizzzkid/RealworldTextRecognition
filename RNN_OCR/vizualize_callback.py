import numpy as np
import keras.callbacks
import pylab
import os
import itertools

# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    keys = [26,27,28,39,30,31,32,33,34,35]
    values = [str(x) for x in range(0,10)]
    dictionary = dict(zip(keys,values))

    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        # 26 is space, 27 is CTC blank char
        outstr = ''
        for c in out_best:
            if c >= 0 and c < 26:
                outstr += chr(c + ord('a'))
            elif c == 36:
                outstr += ' '
            elif c >=26 and c <=35:
                outstr += dictionary[c]
        ret.append(outstr)
    return ret



class VizCallback(keras.callbacks.Callback):

    def __init__(self, test_func, text_img_gen, num_display_words=6):
        self.test_func = test_func
        self.output_dir = os.path.join('results')
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_epoch_end(self, epoch, logs={}):
        pass
        # print("Epoch Ending -> Exporting model and validation results...")
        # self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
        
        # word_batch = next(self.text_img_gen)[0]
        # res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
        # if word_batch['the_input'][0].shape[0] < 256:
        #     cols = 2
        # else:
        #     cols = 1
        # for i in range(self.num_display_words):
        #     pylab.subplot(self.num_display_words // cols, cols, i + 1)
        #     if K.image_data_format() == 'channels_first':
        #         the_input = word_batch['the_input'][i, 0, :, :]
        #     else:
        #         the_input = word_batch['the_input'][i, :, :, 0]
        #     pylab.imshow(the_input.T, cmap='Greys_r')
        #     pylab.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
        # fig = pylab.gcf()
        # fig.set_size_inches(10, 13)
        # pylab.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
        # pylab.close()