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
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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


def getPredictedValue(model,documentVector,predictInput):
    model.load()
    inputVector = 0
    outputVector = 0
    isFirst = True
    predictedLabels = []
    element = model.predict(predictInput)
    for doc in documentVector:
        # print(doc)
        if isFirst is True:
            outputVector = int(doc)
            isFirst = False
            # print("FIRST ELEMENT IS OFF")
        else:
            outputVector = inputVector + int(doc) + 1
            # print("SECOND ELEMENT")

        isZero = not np.count_nonzero(element[inputVector:outputVector])
        if isZero is True: # if no target can be found 
            predictedLabels.append(1) # # mark it as "OUTSIDE"
        else:
            predictedLabels.append(0) # if it is 'INSIDE'
        inputVector = outputVector
    print(predictedLabels)
    return predictedLabels





if __name__ == '__main__':
    # locale.getpreferredencoding = getpreferredencoding a
    saveFolder = './data/output'
    filePath = './raw_data/alta/test_alta_dataset.csv'
    # nlp = spacy.load("en_core_web_sm") # load our spacy model for it
    # nlp.tokenizer = Tokenizer(nlp.vocab)
    # #Checking if we can preprocess them properly and then load our model to check if it would work or not
    praw.process_pandas2(filePath, is_train_file=False, save_folder=saveFolder , isClean=True, countSentence=True) # this will process raw
    glove_vectors, glove_embed_dim = prepro.load_glove_format('./raw_data/glove.42B.300d.txt') # load the embeddings
#    prepro.process_predict(saveFolder, lambda x: prepro.spacyTokenizer_train(x,True,True), True) # this would do the pre_processing for the data to predict
    config = Config() # load our config file
    config.use_elmo = True
    config.use_elmo_alone = True
    config.elmo_trainable = False
    config.word_embed_trainable = True
    config.aspect_embed_trainable = True
#    model = loadModel('alta2', 'twitter', 'word', 'td_lstm') # pick when model to load and to do the test #td_lstm
#    predict_input = load_input_data('output', 'train', config.level, config.use_text_input, config.use_text_input_l, #temp workaround
#                             config.use_text_input_r, config.use_text_input_r_with_pad, config.use_aspect_input,
#                             config.use_aspect_text_input, config.use_loc_input, config.use_offset_input,
#                             config.use_mask)
#    documentVec = np.load(saveFolder+"/totalsentence.npy")
    labels = getPredictedValue(model,documentVec,predict_input)
    np.save(saveFolder+"/predictedval.npy",labels) #added the option to save labels
    # predictValue(model,[26,31],predict_input)
    # element = model.predict(predict_input)
    # print(element[0:25])
    # tester = element[0:26]
    # # print(element)
    # print(collections.Counter(element))
    # print(collections.Counter(element[0:26]))
    # print(collections.Counter(element[26:58]))
    # print(np.count_nonzero(element[0:26]))
    # print(np.count_nonzero(element[26:58]))







	#model = loadModel('books', 'laptop', 'word', 'td_lstm') #pattern for our prediction to load our model
