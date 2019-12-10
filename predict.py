import os
import time
from config import Config
from data_loader import load_idx2token
from data_loader import load_input_data, load_label
from models import SentimentModel
import models
import preprocess as prepro
import process_raw as praw
import spacy
from spacy.tokenizer import Tokenizer
import locale
import collections


##WORKAROUND FOR STUPID PYTHON 3.6 AND THE UTF-8 ISSUE WITH THE SERVER 
def getpreferredencoding(do_setlocale = True):
   return "utf-8"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def loadModel(data_folder, data_name, level, model_name, is_aspect_term=True):
    config.data_folder = data_folder
    config.data_name = data_name
    if not os.path.exists(os.path.join(config.checkpoint_dir, data_folder)):
        os.makedirs(os.path.join(config.checkpoint_dir, data_folder))
    config.level = level
    config.model_name = model_name
    config.is_aspect_term = is_aspect_term
    config.init_input()
    config.exp_name = '{}_{}_wv_{}'.format(model_name, level, config.word_embed_type)
    config.exp_name = config.exp_name + '_update' if config.word_embed_trainable else config.exp_name + '_fix'
    if config.use_aspect_input:
        config.exp_name += '_aspv_{}'.format(config.aspect_embed_type)
        config.exp_name = config.exp_name + '_update' if config.aspect_embed_trainable else config.exp_name + '_fix'
    if config.use_elmo:
        config.exp_name += '_elmo_alone_{}_mode_{}_{}'.format(config.use_elmo_alone, config.elmo_output_mode,
                                                              'update' if config.elmo_trainable else 'fix')

    print(config.exp_name)
    model = SentimentModel(config)
    return model

if __name__ == '__main__':
    # locale.getpreferredencoding = getpreferredencoding
    # nlp = spacy.load("en_core_web_sm") # load our spacy model for it
    # nlp.tokenizer = Tokenizer(nlp.vocab)
    # #Checking if we can preprocess them properly and then load our model to check if it would work or not
    # praw.process_pandas2('./raw_data/alta/train_67.csv', is_train_file=False, save_folder='./data/output' , isClean=True) # this will process raw
    # glove_vectors, glove_embed_dim = prepro.load_glove_format('./raw_data/glove.42B.300d.txt') # load the embeddings
    # prepro.process_predict('./data/output', lambda x: prepro.spacyTokenizer(x), True) # this would do the pre_processing for the data to predict
    config = Config() # load our config file
    config.use_elmo = False
    config.use_elmo_alone = False
    config.elmo_trainable = False
    config.word_embed_trainable = True
    config.aspect_embed_trainable = True
    model = loadModel('alta2', 'twitter', 'word', 'td_lstm') # pick when model to load and to do the test
    predict_input = load_input_data('output', 'train', config.level, config.use_text_input, config.use_text_input_l,
                             config.use_text_input_r, config.use_text_input_r_with_pad, config.use_aspect_input,
                             config.use_aspect_text_input, config.use_loc_input, config.use_offset_input,
                             config.use_mask)
    model.load()
    element = model.predict(predict_input)
    print(element[0:25])
    tester = element[0:25]
    # print(element)
    print(collections.Counter(element))
    print(collections.Counter(tester[0:25]))








	#model = loadModel('books', 'laptop', 'word', 'td_lstm') #pattern for our prediction to load our model