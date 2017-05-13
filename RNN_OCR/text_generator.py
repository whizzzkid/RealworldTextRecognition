import numpy as np
import keras.callbacks

from mjsynth_dictnet import MJSYNTH_DICTNET

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def is_valid_str(in_str):
    search = re.compile(r'[^a-z\ ]').search
    return not bool(search(in_str))

# def text_to_labels(text, num_classes):
#     ret = []
#     values = [26,27,28,39,30,31,32,33,34,35]
#     keys = [str(x) for x in range(0,10)]
#     dictionary = dict(zip(keys,values))
#     for char in text:       
#         if char >= 'a' and char <= 'z':
#             ret.append(ord(char) - ord('a'))
#         elif char == ' ':
#             ret.append(36)
#         elif char >='0' and char <='9':
#             ret.append(dictionary[char])
#         else:
#             print("error on:", text)    
#     ret.append(37)
#     return ret
def text_to_labels(text, num_classes):
    ret = []
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
        else:
            ret.append(ord('a') - ord('a'))
    ret.append(27)
    return ret

class TextGenerator(keras.callbacks.Callback):

    def __init__(self, minibatch_size,
                 img_w, img_h, downsample_factor, lexicon,
                 valid_class, valid_examples,
                 absolute_max_string_len=23):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h      
        self.downsample_factor = downsample_factor        
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len
        self.lexicon = lexicon
        self.previous_examples = []   
        self.num_train_words = 0    
        self.valid_class = valid_class
        self.valid_examples = valid_examples 

    def get_output_size(self):
        return 38

    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words, examples_per_class): 
        print("Building Word lists...")       
        self.synth_load_train = MJSYNTH_DICTNET("train",num_words,examples_per_class,self.previous_examples)
        print("Word List Built: size =", len(self.synth_load_train.labels))
        self.synth_load_valid = MJSYNTH_DICTNET("valid",self.valid_class,self.valid_examples,[])
        print("Valid Word List Build: size =", len(self.synth_load_valid.labels))
        
        self.previous_examples = self.synth_load_train.class_mapping + self.previous_examples

        self.Y_data = np.ones([num_words*examples_per_class, self.absolute_max_string_len]) * -1        
        self.X_text = []
        self.Y_len = [0] * (num_words*examples_per_class)   

        for i,y in enumerate(self.synth_load_train.labels):
            word = self.lexicon[int(self.synth_load_train.classes[y[0]])]
            self.Y_len[i] = len(word)
            self.Y_data[i, 0:len(word)+1] = text_to_labels(word, self.get_output_size())
            self.X_text.append(word)       
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

        self.Y_data_valid = np.ones([num_words*examples_per_class, self.absolute_max_string_len]) * -1        
        self.X_text_valid = []
        self.Y_len_valid = [0] * (num_words*examples_per_class)   

        for i,y in enumerate(self.synth_load_valid.labels):
            word = self.lexicon[int(self.synth_load_valid.classes[y[0]])]
            self.Y_len_valid[i] = len(word)
            self.Y_data_valid[i, 0:len(word)+1] = text_to_labels(word, self.get_output_size())
            self.X_text_valid.append(word)       
        self.Y_len_valid = np.expand_dims(np.array(self.Y_len), 1)

        self.cur_train_index = 0
        self.cur_val_index = 0        
        self.num_train_words = num_words
        self.num_valid_words = self.valid_class * self.valid_examples
        

    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
       
        X_data = np.ones([size, self.img_w, self.img_h, 1])
        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []

        for i in range(0, size):   
            if(train):                  
                X_data[i, 0:self.img_w, :, 0] = self.synth_load_train.x[index+i,:,:,0].T
                labels[i, :] = self.Y_data[index + i]
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = self.Y_len[index + i]
                source_str.append(self.X_text[index + i])
            else:
                X_data[i, 0:self.img_w, :, 0] = self.synth_load_valid.x[index+i,:,:,0].T
                labels[i, :] = self.Y_data_valid[index + i]
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = self.Y_len_valid[index + i]
                source_str.append(self.X_text_valid[index + i])

        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.num_train_words:
                self.cur_train_index = 0            
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_valid_words:
                self.cur_val_index = 0
            yield ret

    def on_train_begin(self, logs={}):
        self.build_word_list(320, 1)       

    def on_epoch_begin(self, epoch, logs={}):
        pass
        

if __name__ == '__main__':
    #Tests
    lexicon = np.genfromtxt('../data/mnt/ramdisk/max/90kDICT32px/lexicon.txt', dtype='str' )
    img_gen = TextGenerator(minibatch_size=32,
                                 img_w=100,
                                 img_h=32,
                                 downsample_factor=4, 
                                 valid_class = 32,
                                 valid_examples = 2,    
                                 lexicon = lexicon                            
                                 )
    img_gen.build_word_list(32,2)
    print("Total training images loaded: ", len(img_gen.X_text))
    print("Total validation images loaded: ", len(img_gen.X_text_valid))

    print("Getting Training Batch")
    gen = img_gen.next_train()
    (inputs, outputs) = next(gen)
    print(inputs['the_input'].shape)

    z = inputs['the_input']
    gs = gridspec.GridSpec(8,4, top=1., bottom=0., right=1., left=0., hspace=0.1, wspace=0.1)
    for i,g in enumerate(gs): 
        ax = plt.subplot(g)
        ax.imshow(z[i,:,:,0].T, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.set_title("Class: " + inputs['source_str'][i])
    plt.title("Mini-Batch Images")
    plt.show()

    print("Getting Validation Batch")
    gen = img_gen.next_val()
    (inputs, outputs) = next(gen)
    print(inputs['the_input'].shape)

    z = inputs['the_input']
    gs = gridspec.GridSpec(8,4, top=1., bottom=0., right=1., left=0., hspace=0.1, wspace=0.1)
    for i,g in enumerate(gs): 
        ax = plt.subplot(g)
        ax.imshow(z[i,:,:,0].T, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.set_title("Class: " + inputs['source_str'][i])
    plt.title("Mini-Batch Images (valid)")
    plt.show()

    print("\n-------Testing Previous Examples---------")
    print(img_gen.previous_examples)
    img_gen.build_word_list(32,2)
    print(img_gen.previous_examples)

  