

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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



# 1. MNLI by genre - RNN
pred_mnli = pickle.load(open('../model/mnli_eval/_pred.p', 'rb'))
df_mnli = pd.DataFrame.from_dict({'Y_hat_class': pred_mnli[1],
                                       'Y': pred_mnli[2], 'genre': pred_mnli[3]})

df_mnli.groupby('genre').apply(lambda x: sum(x['Y_hat_class'] == x['Y']) / len(x))

# 2. MNLI by genre - CNN
pred_mnli_cnn = pickle.load(open('../model/Mnli_CNN_eval/_pred.p', 'rb'))
df_mnli_cnn = pd.DataFrame.from_dict({'Y_hat_class': pred_mnli_cnn[1],
                                       'Y': pred_mnli_cnn[2], 'genre': pred_mnli_cnn[3]})

df_mnli_cnn.groupby('genre').apply(lambda x: sum(x['Y_hat_class'] == x['Y']) / len(x))



# 3. Hyper-parameter - CNN, Filter size
dir = '../model/CNN/K3_5_filters_'
lsResult_CNNF = []
for i in range(1,7):
    lsResult_CNNF.append(pickle.load(open(dir + str(i) + '/_accuracy.p', 'rb')))
lsNames = ['F1: 512, F2: 256, L1: 128', 'F1: 256, F2: 256, L1: 128', 'F1: 256, F2: 128, L1: 64',
           'F1: 128, F2: 64, L1: 64',  'F1: 64, F2: 64, L1: 32', 'F1: 64, F2: 32, L1: 16']
lsShape = ['c-', 'm-', 'b-', 'k-', 'r-', 'g-', 'r-']

plot(lsResult_CNNF, lsNames, lsShape, outPath='../plots/CNN_F')

# 4. Hyper-parameter - CNN, Kernel size
dir = '../model/CNN/F256_256_128_Ks_'
lsResult_CNNK = []
for i in [1, 3, 7, 8, 9, 11]:
    lsResult_CNNK.append(pickle.load(open(dir + str(i) + '/_accuracy.p', 'rb')))
lsNames = ['K1: 2, K2: 2', 'K1: 3, K2: 3', 'K1: 4, K2: 4',
           'K1: 5, K2: 3',  'K1: 5, K2: 5', 'K1: 7, K2: 5']
lsShape = ['c-', 'm-', 'b-', 'k-', 'r-', 'g-', 'r-']

plot(lsResult_CNNK, lsNames, lsShape, outPath='../plots/CNN_K')

# 5. Hyper-parameter - CNN, product v.s. concat
dir = ['../model/CNN/F256_256_128_Ks_1', '../model/CNN/Prod_F256_256_128_Ks_1']
lsResult_CNNProd = []
for d in dir:
    lsResult_CNNProd.append(pickle.load(open(d + '/_accuracy.p', 'rb')))
lsNames = ['Concatenate', 'Elementwise Product']
lsShape = ['m-', 'b-']

plot(lsResult_CNNProd, lsNames, lsShape, outPath='../plots/CNN_prod')


# 6. Hyper-parameter - RNN, LSTM dim
dir = '../model/RNN/Prod_dimLSTM_'
lsResult_RNNdim = []
for i in range(1,7):
    lsResult_RNNdim.append(pickle.load(open(dir + str(i) + '/_accuracy.p', 'rb')))
lsNames = ['dimLSTM: 256, L1: 512', 'dimLSTM: 256, L1: 256', 'dimLSTM: 128, L1: 256',
           'dimLSTM: 128, L1: 128',  'dimLSTM: 64, L1: 128', 'dimLSTM: 64, L1: 64']
lsShape = ['c-', 'm-', 'b-', 'k-', 'r-', 'g-', 'r-']

plot(lsResult_RNNdim, lsNames, lsShape, outPath='../plots/RNNdim')



# 7. Hyper-parameter - RNN, product v.s. concat
dir = ['../model/RNN/dimLSTM_3', '../model/RNN/Prod_dimLSTM_3']
lsResult_RNNProd = []
for d in dir:
    lsResult_RNNProd.append(pickle.load(open(d + '/_accuracy.p', 'rb')))
lsNames = ['Concatenate', 'Elementwise Product']
lsShape = ['m-', 'b-']

plot(lsResult_RNNProd, lsNames, lsShape, outPath='../plots/RNN_prod')



# 8. Prediction examples
pred_snli = pickle.load(open('../model/RNN/Prod_dimLSTM_Save_3/_pred.p', 'rb'))
pred_snli2 = [x for x in zip(pred_snli[0], pred_snli[1], pred_snli[2])]