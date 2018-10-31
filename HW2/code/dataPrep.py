'''
Load embedding, create dictionary, convert text to index
'''


import io
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import argparse
#import json
import os
import numpy as np
import pickle
import pdb

def text2index(text, vocab, analyzer):
    # 1 is unk
    doc_toks = [vocab[y] if y in vocab else 1 for y in analyzer(text) ]
    return doc_toks

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.fromiter(map(float, tokens[1:]), dtype=np.float)
    return data


def build_vocab(text, emb, emb_dim=300, max_df=.7, max_features=20000, stop_words= 'english'):
    '''
    Fit vocabulary
    :param text: list of documents for creating vocabulary
    :return: vectorizer
    '''
    vect = CountVectorizer(stop_words=stop_words, max_df=max_df, max_features=max_features,
                           token_pattern=r"(?u)[!\"#\$\%&\'()\*\+,-./:;<=>\?@\[\\\]\^_`{|}~\w]+")
    vect.fit(text)

    no_embedding = [k for k in vect.vocabulary_.keys() if k not in emb]
    print("No Embeddings for: ")
    print(len(no_embedding))

    vocab = [k for i, k in enumerate(vect.vocabulary_.keys()) if k in emb]
    new_vocab = dict([(k, i + 2) for i, k in enumerate(vocab)])
    # Set 0 to be the padding index, 1 to be unk
    vect.vocabulary_ = new_vocab
    print('Vocabulary size: ', len(new_vocab))

    embedding = np.zeros(shape=(len(new_vocab) + 2, emb_dim))
    for k,i in new_vocab.items():
        embedding[i] = emb[k]

    return vect, embedding

def df2List(df, vocab, analyzer, label_dict, ismnli = False):
    out = []
    for i, row in df.iterrows():
        set1 = text2index(row['sentence1'], vocab, analyzer)
        set2 = text2index(row['sentence2'], vocab, analyzer)
        label = label_dict[row['label']]
        if ismnli:
            genre = row['genre']
        else:
            genre = 'snli'
        out.append([set1, set2, label, i, genre])
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputPath", default = '../hw2_data/') # Should have train/val in this directory
    parser.add_argument("--embPath", default='../hw2_data/wiki-news-300d-1M.vec')  # embedding vector path
    parser.add_argument("--emb_dim", type=int, default = 300)
    parser.add_argument("--outPath")  # Output Path
    parser.add_argument("--max_df", type=float, default = 0.7)
    parser.add_argument("--max_features", type=int, default=20000)
    parser.add_argument("--stop_words", default = 'english')

    args = parser.parse_args()

    if not os.path.isdir(args.outPath):
        os.mkdir(args.outPath)

    print("Data processing parameters: ", args)

    print("Loading Data")
    train = pd.read_csv(args.inputPath + 'snli_train.tsv', header = 0, sep = '\t')
    test = pd.read_csv(args.inputPath + 'snli_val.tsv', header=0, sep='\t')
    train_mnli = pd.read_csv(args.inputPath + 'mnli_train.tsv', header=0, sep='\t')
    test_mnli = pd.read_csv(args.inputPath + 'mnli_val.tsv', header=0, sep='\t')


    
    emb = load_vectors(args.embPath)

    print("Fitting Vocabulary")
    vect, embedding = build_vocab(train['sentence1'] + ' ' + train['sentence2'], emb, emb_dim = args.emb_dim,
                                   max_df = args.max_df, max_features = args.max_features, stop_words=args.stop_words)

    #vect = pickle.load(open(args.outPath + 'vect.p', 'rb'))

    vocab = vect.vocabulary_
    analyzer = vect.build_analyzer()

    print('Transform data frame')
    label_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    train2 = df2List(train, vocab, analyzer, label_dict)
    test2 = df2List(test, vocab, analyzer, label_dict)
    train_mnli2 = df2List(train_mnli, vocab, analyzer, label_dict, ismnli = True)
    test_mnli2 = df2List(test_mnli, vocab, analyzer, label_dict, ismnli = True)

    pickle.dump(train2, open(args.outPath + 'train.p', 'wb'))
    pickle.dump(test2, open(args.outPath + 'test.p', 'wb'))
    pickle.dump(train_mnli2, open(args.outPath + 'train_mnli.p', 'wb'))
    pickle.dump(test_mnli2, open(args.outPath + 'test_mnli.p', 'wb'))
    pickle.dump(vect, open(args.outPath + 'vect.p', 'wb'))
    pickle.dump(embedding, open(args.outPath + 'embedding.p', 'wb'))

    # Document length:

    lsLen = [max(len(x[0]), len(x[1])) for x in train2]
    print('Median doc size: ', np.percentile(lsLen, 50))
    print('95 percentile: ', np.percentile(lsLen, 95))
    print('Max: ', max(lsLen))

    lsLen = [max(len(x[0]), len(x[1])) for x in train_mnli2]
    print('Median mnli_doc size: ', np.percentile(lsLen, 50))
    print('95 percentile: ', np.percentile(lsLen, 95))
    print('Max: ', max(lsLen))
