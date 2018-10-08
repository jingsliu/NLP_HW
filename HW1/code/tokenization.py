from sklearn.feature_extraction.text import CountVectorizer
import argparse
import json
import os
import numpy as np

# Script to tokenize text

def text2index(text, vocab, tokenizer):
    # words not in vocab are dropped
    doc_toks = [vocab[y] for y in tokenizer(text) if y in vocab]
    return doc_toks

def build_vocab(text, max_df=.7, max_features=20000, stop_words= 'english', analyzer = 'word', ngram_range=(1, 1)):
    '''
    Fit vocabulary
    :param text: list of documents for creating vocabulary
    :return: vectorizer
    '''
    vect = CountVectorizer(stop_words=stop_words, max_df=max_df, max_features=max_features,
                           analyzer=analyzer, ngram_range=ngram_range)
    vect.fit(text)
    new_vocab = dict([(k, i + 1) for i, k in enumerate(vect.vocabulary_.keys())])  # Set 0 to be the padding index
    vect.vocabulary_ = new_vocab
    print('Vocabulary size: ', len(new_vocab))
    return vect

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputPath", default = '../aclImdb/') # Should have train/dev/test in this directory
    parser.add_argument("--outPath")  # Output Path
    parser.add_argument("--max_df", type=float, default = 0.7)
    parser.add_argument("--max_features", type=int, default=20000)
    parser.add_argument("--stop_words", default = 'english')
    parser.add_argument("--analyzer", default = 'word')
    parser.add_argument("--ngramMax", type = int, default=1)
    parser.add_argument("--ngramMin", type=int, default=1)

    args = parser.parse_args()

    if not os.path.isdir(args.outPath):
        os.mkdir(args.outPath)

    print("Data processing parameters: ", args)

    print("Loading Data")
    train = json.load(open(args.inputPath + 'train.json', 'r'))
    val = json.load(open(args.inputPath + 'val.json', 'r'))
    test = json.load(open(args.inputPath + 'test.json', 'r'))

    print("Fitting Vocabulary")
    vect = build_vocab([x[0] for x in train], max_df = args.max_df, max_features = args.max_features,
                       stop_words=args.stop_words, analyzer=args.analyzer,
                       ngram_range=(args.ngramMin, args.ngramMax))
    vocab = vect.vocabulary_
    tokenizer = vect.build_tokenizer()

    train2 = [[text2index(x[0], vocab, tokenizer), x[1], x[2]] for x in train]
    val2 = [[text2index(x[0], vocab, tokenizer), x[1], x[2]] for x in val]
    test2 = [[text2index(x[0], vocab, tokenizer), x[1], x[2]] for x in test]

    json.dump(train2, open(args.outPath + 'train.json', 'w'))
    json.dump(val2, open(args.outPath + 'val.json', 'w'))
    json.dump(test2, open(args.outPath + 'test.json', 'w'))

    # Document length:
    lsLen = [len(x[0]) for x in train]
    print('Median doc size: ', np.percentile(lsLen, 50))
    print('95 percentile: ', np.percentile(lsLen, 95))
    print('Max: ', max(lsLen))