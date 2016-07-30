__author__ = 'ccqy'
# -*- coding:utf-8 -*-

import jieba
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from clean import clean_train, clean_test
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def words_to_features(raw_line, stopwords_path='./stop.txt'):
    stopwords = {}.fromkeys(
        [line.rstrip() for line in open(stopwords_path)]
    )
    chinese_only = raw_line
    words_lst = jieba.cut(chinese_only)
    meaninful_words = []
    for word in words_lst:
        word = word.encode('utf8')
        if word not in stopwords:
            meaninful_words.append(word)


    return ' '.join(meaninful_words)


def drawfeature(train_data_path='./train', train_filename='train_cleaned',test_data_path='./test', test_filename='test_cleaned'):
    train_file = os.path.join(train_data_path, train_filename)
    train_data = pd.read_csv(train_file)
    n_train_data = train_data['text'].size

    test_file = os.path.join(test_data_path,test_filename)
    test_data = pd.read_csv(test_file)
    n_test_data = test_data['text'].size

    vectorizer = CountVectorizer(analyzer="word",tokenizer=None, preprocessor=None, stop_words=None, max_features=2000)
    transformer = TfidfTransformer()

    train_data_words = []
    for i in xrange(n_train_data):
        train_data_words.append(words_to_features(train_data['text'][i]))
    train_data_features = vectorizer.fit_transform(train_data_words)
    train_data_features = train_data_features.toarray()
    train_data_features = transformer.fit_transform(train_data_features)
    train_data_features = train_data_features.toarray()
    train_data_pd=pd.Series(train_data_features,name=None)
    train_data_pd.to_csv("trainfeature.csv", index=None, header=True)


    test_data_words = []
    for i in xrange(n_test_data):
        test_data_words.append(words_to_features(test_data['text'][i]))
    test_data_features = vectorizer.fit_transform(test_data_words)
    test_data_features = test_data_features.toarray()
    test_data_features = transformer.fit_transform(test_data_features)
    test_data_features = test_data_features.toarray()
    test_data_pd=pd.Series(test_data_features,name=None)
    test_data_pd.to_csv("testfeature.csv", index=None, header=True)

    forest = RandomForestClassifier(n_estimators=60)
    forest = forest.fit(train_data_features, train_data['lable'])
    pred = forest.pedict(test_data_features)
    pred = pd.Series(pred,name='Target')
    pred.to_csv("bow_tfidf_RF.csv", index=None, header=True)

drawfeature()
