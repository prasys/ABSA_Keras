# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: train.py

@time: 2019/1/5 10:02

@desc:

"""

import os
import time
from config import Config
from data_loader import load_input_data, load_label
from models import SentimentModel
from sklearn.utils import resample # to handle resampling technique to resample the minority class to see if it works

os.environ['CUDA_VISIBLE_DEVICES'] = '0'




def train_model(data_folder, data_name, level, model_name, is_aspect_term=True,classWeights=None):
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

    test_input = load_input_data(data_folder, 'test', level, config.use_text_input, config.use_text_input_l,
                                 config.use_text_input_r, config.use_text_input_r_with_pad, config.use_aspect_input,
                                 config.use_aspect_text_input, config.use_loc_input, config.use_offset_input,
                                 config.use_mask)
    test_label = load_label(data_folder, 'test')

    if not os.path.exists(os.path.join(config.checkpoint_dir, '%s/%s.hdf5' % (data_folder, config.exp_name))):
        start_time = time.time()

        train_input = load_input_data(data_folder, 'train', level, config.use_text_input, config.use_text_input_l,
                                      config.use_text_input_r, config.use_text_input_r_with_pad,
                                      config.use_aspect_input, config.use_aspect_text_input, config.use_loc_input,
                                      config.use_offset_input, config.use_mask)
        train_label = load_label(data_folder, 'train')
        valid_input = load_input_data(data_folder, 'valid', level, config.use_text_input, config.use_text_input_l,
                                      config.use_text_input_r, config.use_text_input_r_with_pad,
                                      config.use_aspect_input, config.use_aspect_text_input, config.use_loc_input,
                                      config.use_offset_input, config.use_mask)
        valid_label = load_label(data_folder, 'valid')

        '''
        Note: Here I combine the training data and validation data together, use them as training input to the model, 
              while I use test data to server as validation input. The reason behind is that i want to fully explore how 
              well can the model perform on the test data (Keras's ModelCheckpoint callback can help usesave the model 
              which perform best on validation data (here the test data)).
              But generally, we won't do that, because test data will not (and should not) be accessible during training 
              process.
        '''
        train_combine_valid_input = []
        for i in range(len(train_input)):
            train_combine_valid_input.append(train_input[i] + valid_input[i])
        train_combine_valid_label = train_label + valid_label
        model.train(train_combine_valid_input, train_combine_valid_label, test_input, test_label,classWeights)
        # model.train(train_combine_valid_input, train_combine_valid_label, test_input, test_label)

        elapsed_time = time.time() - start_time
        print('training time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # load the best model
    model.load()

    # print('score over valid data...')
    # model.score(valid_input, valid_label)
    print('score over test data...')
    model.score(test_input, test_label)


if __name__ == '__main__':
    config = Config()
    config.use_elmo = False
    config.use_elmo_alone = False
    config.elmo_trainable = False

    config.word_embed_trainable = True
    config.aspect_embed_trainable = True
    class_weight = {0 : 1, 1 : 6.5} #increase the rate
    train_model(data_folder='alta2', data_name='twitter', level='word', model_name='td_lstm',is_aspect_term=True,classWeights=class_weight)
    # train_model('alta2', 'twitter', 'word', 'tc_lstm')
    # train_model('alta2', 'twitter', 'word', 'ae_lstm')
    # train_model('alta2', 'twitter', 'word', 'at_lstm')
    # train_model('alta2', 'twitter', 'word', 'atae_lstm')
   # train_model('alta2', 'twitter', 'word', 'memnet')
   # train_model('alta2', 'twitter', 'word', 'ram')
   # train_model('alta2', 'twitter', 'word', 'ian')
   # train_model('alta2', 'twitter', 'word', 'cabasc')

