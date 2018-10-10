import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import pdb
import json

def plot(lsResults, lsNames, lsShape, outPath, legendPos = (0.6, 0.1, 1., .3), ylim=[0,100]):
    lsTrain = [x[1] for x in lsResults]
    lsTest = [x[2] for x in lsResults]
    lsEpoch = [x[0] for x in lsResults]
    maxEpoch = max([x[-1] for x in lsEpoch])
    figure1 = plt.figure(figsize=(16, 6))
    gs1 = gridspec.GridSpec(1, 2)

    ax1 = figure1.add_subplot(gs1[0])
    for i in range(len(lsResults)):
        ax1.plot(np.asarray(lsEpoch[i]), np.asarray(lsTrain[i]), lsShape[i], label=lsNames[i])
    plt.legend(bbox_to_anchor=legendPos, loc=len(lsTrain), ncol=1, borderaxespad=0.)
    ax1.set_xlim([0, maxEpoch + 1])
    ax1.set_ylim(ylim)
    ax1.set_ylabel('Training accuracy', fontsize=12)
    ax1.set_xlabel('Number of Epochs', fontsize=12)
    ax1.set_title('Training accuracy by number of epochs', fontsize=16)

    ax2 = figure1.add_subplot(gs1[1])
    for i in range(len(lsResults)):
        ax2.plot(np.asarray(lsEpoch[i]), np.asarray(lsTest[i]), lsShape[i], label=lsNames[i])
    plt.legend(bbox_to_anchor=legendPos, loc=len(lsTest), ncol=1, borderaxespad=0.)
    ax2.set_xlim([0, maxEpoch + 1])
    ax2.set_ylim(ylim)
    ax2.set_ylabel('Validation accuracy', fontsize=12)
    ax2.set_xlabel('Number of Epochs', fontsize=12)
    ax2.set_title('Validation accuracy by number of epochs', fontsize=16)

    figure1.savefig(outPath + '_plot.png')
    plt.close(figure1)



#==== 1. Initial learning rate ===========

dir = '../model/df07f20K_stopEng_W_1gram/doc300Emb100Vocab20K_AdamLrdecayNone_testLR_'
lsResult_lr = []
for i in range(1,7):
    lsResult_lr.append(pickle.load(open(dir + str(i) + '/_accuracy.p', 'rb')))
lsNames = ['lr=0.1', 'lr=1e-3', 'lr=1e-4', 'lr=1e-5', 'lr=1.0', 'lr=1e-2']
lsShape = ['c-', 'm-', 'b-', 'k-', 'r-', 'g-']

plot(lsResult_lr, lsNames, lsShape, outPath='../plots/AdamLR')

#==== 2. Adam v.s. SGD ========
dir = '../model/df07f20K_stopEng_W_1gram/doc300Emb100Vocab20K_SGDLrdecayNone_testLR_'

lsResult_lrSGD = []
for i in range(1,7):
    lsResult_lrSGD.append(pickle.load(open(dir + str(i) + '/_accuracy.p', 'rb')))
lsNames = ['lr=0.1', 'lr=1e-3', 'lr=1e-4', 'lr=1e-5', 'lr=1.0', 'lr=1e-2']
lsShape = ['c-', 'm-', 'b-', 'k-', 'r-', 'g-']
plot(lsResult_lrSGD, lsNames, lsShape, outPath='../plots/SGDLR')


#==== 3. Annealing =====
dir = '../model/df07f20K_stopEng_W_1gram/doc300Emb100Vocab20K_AdamLR0.01_testDecay_'
lsResult_lrDecay = [lsResult_lr[-1]]
for i in range(1,5):
    lsResult_lrDecay.append(pickle.load(open(dir + str(i) + '/_accuracy.p', 'rb')))
lsNames = ['Constant LR', 'Linear decay (0.01, 1)', 'Linear decay (0.05, 10)',
           'Exponential decay (0.9, 1)', 'Exponential decay (0.9, 10)']
lsShape = ['g-', 'c-', 'r-', 'b-', 'k-']
plot(lsResult_lrDecay, lsNames, lsShape, outPath='../plots/LRdecay',
     legendPos = (0.4, 0.1, 0.3, 0.3), ylim=[50,100])


#==== 4. Embedding size ======
dir = '../model/df07f20K_stopEng_W_1gram/doc300Vocab20K_AdamLR0.01ExpDecay_testEmbdim_'
lsResult_emb = []
for i in range(1,6):
    lsResult_emb.append(pickle.load(open(dir + str(i) + '/_accuracy.p', 'rb')))
lsNames = ['50', '100', '200', '300', '400']
lsShape = ['g-', 'c-', 'r-', 'b-', 'k-']
plot(lsResult_emb, lsNames, lsShape, outPath='../plots/embdim', legendPos = (0.6, 0.1, .3, .3), ylim=[50,100])

#===== 5. N-gram size, vocab size =====
dir = ['df07f20K_stopEng_W_1gram/doc300Vocab20K_AdamLR0.01ExpDecay_embdim50_1',
       'df07f20K_stopEng_W_2gram/doc300Vocab20K_AdamLR0.01ExpDecay_embdim50_2',
       'df07f20K_stopEng_W_3gram/doc300Vocab20K_AdamLR0.01ExpDecay_embdim50_3',
       'df07f100K_stopEng_W_1gram/doc300Vocab100K_AdamLR0.01ExpDecay_embdim50_1',
       'df07f100K_stopEng_W_2gram/doc300Vocab100K_AdamLR0.01ExpDecay_embdim50_2',
       'df07f100K_stopEng_W_3gram/doc300Vocab100K_AdamLR0.01ExpDecay_embdim50_3',
       'df07f50K_stopEng_W_1_3gram/doc300Vocab50K13Gram_AdamLR0.01ExpDecay_embdim50_1']
lsResult_ngram = []
for i in dir:
    lsResult_ngram.append(pickle.load(open('../model/' + i + '/_accuracy.p', 'rb')))
lsNames = ['1gram, 20K', '2gram, 20K', '3gram, 20K', '1gram, 100K', '2gram, 100K', '3gram, 100K', '1to3gram, 50K']
lsShape = ['g-', 'c-', 'r-', 'b-', 'k-', 'm-', 'y-']
plot(lsResult_ngram, lsNames, lsShape, outPath='../plots/Ngram', legendPos = (0.4, 0.1, .3, .3), ylim=[50,100])

#===== 6. Doc length ======
dir = ['df07f50K_stopEng_W_1_3gram/doc300Vocab50K13Gram_AdamLR0.01ExpDecay_embdim50_1',
       'df07f50K_stopEng_W_1_3gram/doc350Vocab50K13Gram_AdamLR0.01ExpDecay_embdim50_1']

lsResult_doc = []
for i in dir:
    lsResult_doc.append(pickle.load(open('../model/' + i + '/_accuracy.p', 'rb')))
lsNames = ['Length 300', 'Length 350']
lsShape = ['r-', 'b-']
plot(lsResult_doc, lsNames, lsShape, outPath='../plots/doc', legendPos = (0.6, 0.1, .3, .3), ylim=[50,100])


#====== 7. word v.s. char =====
dir = ['df07f50K_stopEng_W_1_3gram/doc350Vocab50K13Gram_AdamLR0.01ExpDecay_embdim50_1',
       'df07f20K_stopEng_C_3gram/doc3KVocab20K_Char_AdamLR0.01ExpDecay_embdim50_3',
       'df07f20K_stopEng_C_4gram/doc3KVocab20K_Char_AdamLR0.01ExpDecay_embdim50_4',
       'df07f20K_stopEng_C_5gram/doc3KVocab20K_Char_AdamLR0.01ExpDecay_embdim50_5',
       'df07f20K_stopEng_C_6gram/doc3KVocab20K_Char_AdamLR0.01ExpDecay_embdim50_6'
       ]

lsResult_wc = []
for i in dir:
    lsResult_wc.append(pickle.load(open('../model/' + i + '/_accuracy.p', 'rb')))
lsNames = ['Word 1-3 gram', 'Char 3gram', 'Char 4gram', 'Char 5gram', 'Char 6gram']
lsShape = ['r-', 'b-', 'g-', 'c-','m-']
plot(lsResult_wc, lsNames, lsShape, outPath='../plots/wc', legendPos = (0.4, 0.1, .3, .3), ylim=[50,100])


#====== 8. Example =====
pred = pickle.load(open('../model/df07f50K_stopEng_W_1_3gram/doc350Vocab50K13Gram_AdamLR0.01ExpDecay_embdim50Save_1/_pred.p','rb'))
pred2 = [x for x in zip(pred[0], pred[1], pred[2])]
for i, item in enumerate(pred2):
    if item[1] != item[2]:
        print(i, item)
    if i > 100:
        break

val = json.load(open('../aclImdb/val.json', 'r'))
val[0][2]
val[43][2]
val[38][2]

val[1][2]
val[2][2]
val[10][2]