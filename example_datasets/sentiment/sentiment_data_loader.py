import copy
import os.path
import re
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from utils.logger import LOG


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def tockenize(x_train,y_train,x_val,y_val):
    word_list = []

    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

    # tockenize
    final_list_train,final_list_test = [],[]
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])

    encoded_train = [1 if label =='positive' else 0 for label in y_train]
    encoded_test = [1 if label =='positive' else 0 for label in y_val]
    return final_list_train, np.array(encoded_train),final_list_test, np.array(encoded_test),onehot_dict

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

class SentimentDataLoader:
    def __init__(self, reduced_size=False):
        nltk.download('stopwords')
        self.vocab = None
        self.x_train_pad = None
        self.x_test_pad = None
        self.y_train = None
        self.y_test = None
        self.batch_size = 50

        self.load_data(reduced_size)

    def get_data_loaders(self):
        if self.x_train_pad is None or self.x_test_pad is None or self.y_train is None or self.y_test is None:
            self.load_data()

        x_train_pad_c = copy.deepcopy(self.x_train_pad)
        x_test_pad_c = copy.deepcopy(self.x_test_pad)
        y_train_c = copy.deepcopy(self.y_train)
        y_test_c = copy.deepcopy(self.y_test)

        # create Tensor datasets
        train_data = TensorDataset(torch.from_numpy(x_train_pad_c), torch.from_numpy(y_train_c))
        valid_data = TensorDataset(torch.from_numpy(x_test_pad_c), torch.from_numpy(y_test_c))

        # make sure to SHUFFLE your data
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=self.batch_size)
        return train_loader, valid_loader

    def load_data(self, reduced_size=False):
        LOG.debug('Now loading data...')
        if os.path.exists('example_datasets/sentiment/data/loaded_values.pt'):
            loaded = torch.load('example_datasets/sentiment/data/loaded_values.pt')
            x_train = loaded['x_train']
            self.y_train = loaded['y_train']
            x_test = loaded['x_test']
            self.y_test = loaded['y_test']
            vocab = loaded['vocab']
        else:
            base_csv = 'example_datasets/sentiment/data/IMDB Dataset.csv'
            df = pd.read_csv(base_csv)

            X,y = df['review'].values,df['sentiment'].values
            x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)

            LOG.debug(f'shape of train data is {x_train.shape}')
            LOG.debug(f'shape of test data is {x_test.shape}')

            if reduced_size or True:
                TRAIN_LIMIT = 25000
                TEST_LIMIT = 5000

                x_train = x_train[:TRAIN_LIMIT]
                y_train = y_train[:TRAIN_LIMIT]
                x_test = x_test[:TEST_LIMIT]
                y_test = y_test[:TEST_LIMIT]

            x_train,self.y_train,x_test,self.y_test,vocab = tockenize(x_train,y_train,x_test,y_test)

            to_save = {
                'x_train': x_train,
                'y_train': self.y_train,
                'x_test': x_test,
                'y_test': self.y_test,
                'vocab': vocab
            }
            torch.save(to_save, 'example_datasets/sentiment/data/loaded_values.pt')

        LOG.debug(f'Length of vocabulary is {len(vocab)}')
        self.vocab = vocab

        #we have very less number of reviews with length > 500.
        #So we will consideronly those below it.
        self.x_train_pad = padding_(x_train,500)
        self.x_test_pad = padding_(x_test,500)
        LOG.debug('data loaded')
