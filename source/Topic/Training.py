# from model import *
# from utils import *

import pandas as pd
import pickle
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore', category=Warning)

import argparse


# def model(): #:if __name__ == '__main__':

def main():
    method = "LDA_BERT"
    samp_size = 51000
    ntopic = 10

    # parser = argparse.ArgumentParser(description='contextual_topic_identification tm_test:1.0')

    # parser.add_argument('--fpath', default='/kaggle/working/train.csv')
    # parser.add_argument('--ntopic', default=10,)
    # parser.add_argument('--method', default='TFIDF')
    # parser.add_argument('--samp_size', default=20500)

    # args = parser.parse_args()

    data = documents  # pd.read_csv('/kaggle/working/train.csv')
    data = data.fillna('')  # only the comments has NaN's
    rws = data.abstract
    sentences, token_lists, idx_in = preprocess(rws, samp_size=samp_size)
    # Define the topic model object
    # tm = Topic_Model(k = 10), method = TFIDF)
    tm = Topic_Model(k=ntopic, method=method)
    # Fit the topic model by chosen method
    tm.fit(sentences, token_lists)
    # Evaluate using metrics
    with open("/kaggle/working/{}.file".format(tm.id), "wb") as f:
        pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)

    print('Coherence:', get_coherence(tm, token_lists, 'c_v'))
    print('Silhouette Score:', get_silhouette(tm))
    # visualize and save img
    visualize(tm)
    for i in range(tm.k):
        get_wordcloud(tm, token_lists, i)