import os
import time
from config import Config
from data_loader import load_idx2token
from models import SentimentModel
import preprocess as prepro
import process_raw as praw
import spacy
from spacy.tokenizer import Tokenizer

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
    nlp = spacy.load("en_core_web_sm") # load our spacy model for it
    nlp.tokenizer = Tokenizer(nlp.vocab)

    praw.process_pandas2('./raw_data/books/book_snippet.xlsx', is_train_file=False, save_folder='./data/books' , isClean=True)
    glove_vectors, glove_embed_dim = load_glove_format('./raw_data/glove.42B.300d.txt') # load the embeddings


    config = Config()
	#model = loadModel('books', 'laptop', 'word', 'td_lstm') #pattern for our prediction to load our model