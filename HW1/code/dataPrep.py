#---- Load dataset and split val -----

import os
import json
import numpy as np

def loadData(path, label):
    out = []
    filelist = os.listdir(path)
    for infile in filelist:
        if infile[-3:] == 'txt':
            fileIN = os.path.join(path, infile)
            with open(fileIN, 'r') as f:
                x = f.readline()
            out.append([x, label, infile])
    return out

train_pos = loadData('../aclImdb/train/pos/', label = 1)
train_neg = loadData('../aclImdb/train/neg/', label = 0)
test_pos = loadData('../aclImdb/test/pos/', label = 1)
test_neg = loadData('../aclImdb/test/neg/', label = 0)

np.random.seed(42)
train = train_pos + train_neg
test = test_pos + test_neg

idx = list(range(len(train)))
np.random.shuffle(idx)

n_train = 20000
train2 = [train[i] for i in idx[0:n_train]]
val = [train[i] for i in idx[n_train:]]

json.dump(train2, open('../aclImdb/train.json', 'w'))
json.dump(val, open('../aclImdb/val.json', 'w'))
json.dump(test, open('../aclImdb/test.json', 'w'))