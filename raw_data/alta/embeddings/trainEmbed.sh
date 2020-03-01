#!/bin/bash
# train SCRIPTS FOR ALL


#ALBERT
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv albert albert_base train_albert_base
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv albert albert_large train_albert_large
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv albert albert_xlarge train_albert_xlarge
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv albert albert_xxlarge train_albert_xxlarge


#XLNET
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv xlnet xlnet_large_cased train_xlnet_large_cased
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv xlnet xlnet_large_cased train_xlnet_large_cased

#BERT 
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv bert bert_base_uncased train_bert_base_uncased
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv bert bert_base_cased train_bert_base_cased
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv bert bert_large_uncased train_bert_large_uncased
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv bert bert_large_cased train_bert_large_cased


#ELMO
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv elmo elmo_bi_lm train_elmo_bi_lm

#ELMO
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv ulmfit ulmfit_forward train_ulmfit_forward
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv ulmfit ulmfit_backward train_ulmfit_backward

#USE
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv use use_dan train_use_elm
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv use use_transformer_large train_use_transformer_large

#Word2Vec
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv word2vec google_news_300 train_google_news_300

#fasttext
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv fasttext common_crawl_300 train_common_crawl_300
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv fasttext wiki_news_300 train_wiki_news_300

#glove
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv glove twitter_200 train_twitter_200
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv glove wiki_300 train_wiki_300
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv glove crawl_42B_300 train_crawl_42B_300
#python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/train_alta_dataset.csv glove crawl_840B_300 train_crawl_840B_300
