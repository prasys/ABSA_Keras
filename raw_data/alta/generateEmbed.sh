#!/bin/bash
# TEST SCRIPTS FOR ALL


#ALBERT
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv albert albert_base test_albert_base
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv albert albert_large test_albert_large
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv albert albert_xlarge test_albert_xlarge
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv albert albert_xxlarge test_albert_xxlarge


#XLNET
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv xlnet xlnet_large_cased test_xlnet_large_cased
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv xlnet xlnet_large_cased test_xlnet_large_cased

#BERT 
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv bert bert_base_uncased test_bert_base_uncased
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv bert bert_base_cased test_bert_base_cased
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv bert bert_large_uncased test_bert_large_uncased
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv bert bert_large_cased test_bert_large_cased


#ELMO
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv elmo elmo_bi_lm test_elmo_bi_lm

#ELMO
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv ulmfit ulmfit_forward test_ulmfit_forward
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv ulmfit ulmfit_backward test_ulmfit_backward

#USE
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv use use_dan test_use_elm
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv use use_transformer_large test_use_transformer_large

#Word2Vec
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv word2vec google_news_300 test_google_news_300

#fasttext
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv fasttext common_crawl_300 test_common_crawl_300
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv fasttext wiki_news_300 test_wiki_news_300

#glove
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv glove twitter_200 test_twitter_200
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv glove wiki_300 test_wiki_300
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv glove crawl_42B_300 test_crawl_42B_300
python bertEmbed.py /data/pradeesh/ABSA_Keras/raw_data/alta/test_alta_dataset.csv glove crawl_840B_300 test_crawl_840B_300

