'''
CNN and RNN models
'''

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
import pickle
import pdb


#==== Data ======

class setDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, root_dir, dsName, transform=None):
        self.root_dir = root_dir
        self.ds = pickle.load(open(root_dir + dsName, 'rb'))
        print('Loaded: ', dsName)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Triggered when you call dataset[i]
        """
        set1, set2, labels, ID, genre = self.ds[idx][0:]
        len1 = len(set1)
        len2 = len(set2)
        set1 = np.asarray(set1, dtype='int')
        set2 = np.asarray(set2, dtype='int')
        labels = np.asarray(labels, dtype='int')

        sample = {'set1': set1, 'set2': set2, 'len1': len1, 'len2': len2,
                  'labels': labels, 'ID': ID, 'genre': genre}
        if self.transform:
            sample = self.transform(sample)

        return sample


class padToTensor(object):
    """
    pad sample to make it as long as the given doc_len
    """
    def __init__(self, doc_len):
        self.doc_len = doc_len

    def __call__(self, sample):
        set1, set2, len1, len2, labels, ID, genre = sample['set1'], sample['set2'], sample['len1'], sample['len2'],\
                                                    sample['labels'], sample['ID'], sample['genre']
        # Pad docs
        set1, len1 = self._pad_doc(set1, self.doc_len, len1)
        set2, len2 = self._pad_doc(set2, self.doc_len, len2)

        return {'ID': ID,
                'len1': torch.from_numpy(np.asarray(len1)).long(),
                'len2': torch.from_numpy(np.asarray(len2)).long(),
                'set1': set1,
                'set2': set2,
                'labels': torch.from_numpy(np.asarray(labels)).long(),
                'genre': genre
                }

    def _pad_doc(self, seq, max_len, length):
        padded_seq = torch.zeros(max_len)
        length = max(1, min(length, max_len))
        s = min(max_len, len(seq))
        padded_seq[0:s] = torch.from_numpy(np.asarray(seq[0:s])).long()
        return padded_seq, length

#===== Model ========

class CNN(nn.Module):
    """
    CNN model
    """

    def __init__(self, model_paras, embedding):
        """
        @param vocab_size: size of the vocabulary.
        @param emb_dim: size of the word embedding
        """
        super(CNN, self).__init__()
        self.model_paras = model_paras
        vocab_size = model_paras.get('nVocab', 20000)
        emb_dim = model_paras.get('dimEmb', 300)
        filters1 = model_paras.get('filters1')
        filters2 = model_paras.get('filters2')
        K1 = model_paras.get('K1')
        K2 = model_paras.get('K2')
        L1 = model_paras.get('L1')
        self.flgProd = self.model_paras.get('flgProd', False)

        # pay attention to padding_idx
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embed.weight = nn.Parameter(embedding, requires_grad=False)

        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters1, K1),
                                    nn.Conv1d(filters1, filters2, K2)])
        self.bn_conv = nn.ModuleList([nn.BatchNorm1d(filters1), nn.BatchNorm1d(filters2)])

        if self.flgProd:
            self.linear1 = nn.Linear(filters2, L1)
        else:
            self.linear1 = nn.Linear(2*filters2, L1)
        self.linear2 = nn.Linear(L1, 3)

        self.params = list(self.linear1.parameters()) + list(self.linear2.parameters())
        for c in self.convs:
            self.params += list(c.parameters())

        for b in self.bn_conv:
            self.params += list(b.parameters())

        n = 0
        for param in self.params:
            n_temp = 1
            for i in param.size():
                n_temp *= i
            n += n_temp
        print('Parameter size: ', n)

    def fc_layer(self, x, layer, bn=None):
        flg_bn = self.model_paras.get('flg_bn', False)
        p_dropOut = self.model_paras.get('p_dropOut', 0.5)

        x = layer(x)
        if (flg_bn is True) & (bn is not None):
            x = F.relu(bn(x))
        else:
            x = F.relu(x)
        x = F.dropout(x, p_dropOut)
        return x

    def encoder(self, words):
        emb = self.embed(words)
        emb = emb.transpose(1, 2)

        h_CNN1 = self.fc_layer(emb, self.convs[0], self.bn_conv[0])
        h_CNN2 = self.fc_layer(h_CNN1, self.convs[1], self.bn_conv[1])

        h_CNN_out = F.max_pool1d(h_CNN2, h_CNN2.size(2)).squeeze(2)
        return h_CNN_out

    def forward(self, set1, set2, len1, len2):

        batchsize = set1.shape[0]
        encode1 = self.encoder(set1)
        encode2 = self.encoder(set2)


        if self.flgProd:
            z = encode1 * encode2
        else:
            z = torch.cat([encode1, encode2], dim = 1)
        l1 = self.fc_layer(z, self.linear1)
        out = self.linear2(l1)
        return out




class RNN(nn.Module):

    # RNN model
    def __init__(self, model_paras, embedding):
        super(RNN, self).__init__()
        self.model_paras = model_paras
        self.flg_cuda = model_paras.get('flg_cuda')
        self.flgProd = self.model_paras.get('flgProd', False)

        vocab_size = model_paras.get('nVocab', 20000)
        emb_dim = model_paras.get('dimEmb', 300)
        self.dimLSTM = model_paras.get('dimLSTM', 128)
        self.p_dropOut = model_paras.get('p_dropOut', 0.5)
        L1 = model_paras.get('L1')

        if self.flgProd:
            self.linear1 = nn.Linear(2 * self.dimLSTM, L1)
        else:
            self.linear1 = nn.Linear(4 * self.dimLSTM, L1)
        self.linear2 = nn.Linear(L1, 3)

        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embed.weight = nn.Parameter(embedding, requires_grad=False)

        self.lstm = nn.GRU(emb_dim, self.dimLSTM, 1, batch_first=True, bidirectional=True, dropout=self.p_dropOut)

        self.params = list(self.lstm.parameters()) + list(self.linear1.parameters()) + list(self.linear2.parameters())

        n = 0
        for param in self.params:
            n_temp = 1
            for i in param.size():
                n_temp *= i
            n += n_temp
        print('Parameter size: ', n)

    def fc_layer(self, x, layer, bn=None):
        flg_bn = self.model_paras.get('flg_bn', True)
        x = layer(x)
        if (flg_bn is True) & (bn is not None):
            x = F.relu(bn(x))
        else:
            x = F.relu(x)
        x = F.dropout(x, self.p_dropOut)
        return x

    def init_hidden(self, batchSize):
        if self.flg_cuda:
            return Variable(torch.zeros(2, batchSize, self.dimLSTM)).cuda()
        else:
            return Variable(torch.zeros(2, batchSize, self.dimLSTM))

    def encoder(self, x, length):
        batchSize = x.shape[0]
        # Ref: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/layers.py#L103-L165
        _, idx_sort = torch.sort(length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(length[idx_sort])
        if self.flg_cuda:
            idx_sort = Variable(idx_sort).cuda()
            idx_unsort = Variable(idx_unsort).cuda()
        else:
            idx_sort = Variable(idx_sort)
            idx_unsort = Variable(idx_unsort)
        # Sort x
        embed = self.embed(x)
        embed = embed.index_select(0, idx_sort)
        rnn_input = nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True)

        h0 = self.init_hidden(batchSize=batchSize)
        rnn_out, hidden = self.lstm(rnn_input, h0)
        hidden = torch.cat([hidden[0].squeeze(), hidden[1].squeeze()], 1)
        hidden = hidden.index_select(0, idx_unsort)
        return hidden

    def forward(self, set1, set2, len1, len2):
        encode1 = self.encoder(set1, len1)
        encode2 = self.encoder(set2, len2)
        flgProd = self.model_paras.get('flgProd', False)
        if flgProd:
            z = encode1 * encode2
        else:
            z = torch.cat([encode1, encode2], dim=1)
        l1 = self.fc_layer(z, self.linear1)
        out = self.linear2(l1)
        return out


#====== Training =======
class trainModel(object):
    def __init__(self, train_paras, train_loader, test_loader, model, optimizer):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer

        # self.train_paras = train_paras
        self.n_batch = train_paras.get('n_batch', 1) # Use this to adjust how many batches as one epoch
        self.n_iter = train_paras.get('n_iter', 1)
        self.log_interval = train_paras.get('log_interval', 1)
        self.flg_cuda = train_paras.get('flg_cuda', False)
        # self.max_len = train_paras.get('max_len', 2000) # Max length of input
        self.lr_decay = train_paras.get('lr_decay', None)  # List of 4 numbers: [init_lr, lr_decay_rate, lr_decay_interval, min_lr, decay_type]
        self.flgSave = train_paras.get('flgSave', False)  # Set to true if save model
        self.savePath = train_paras.get('savePath', './')
        #self.alpha_L1 = train_paras.get('alpha_L1', 0.0)  # Regularization coefficient on fully connected weights

        if self.lr_decay:
            assert len(self.lr_decay) == 5  # Elements include: [starting_lr, decay_multiplier, decay_per_?_epoch, min_lr, decay_type]
        self.criterion = torch.nn.CrossEntropyLoss()
        self.cnt_iter = 0

        self.lsTrainAccuracy = []
        self.lsTestAccuracy = []
        self.lsEpochNumber = []
        self.bestAccuracy = 0.0
        self.acc = 0.0

    def run(self):
        for epoch in range(self.n_iter):
            self._train(epoch)
            self._test(epoch)
            pickle.dump([self.lsEpochNumber, self.lsTrainAccuracy, self.lsTestAccuracy], open(self.savePath + '_accuracy.p', 'wb'))
            if self.acc > self.bestAccuracy:
                self.bestAccuracy = self.acc
                if self.flgSave:
                    self._saveModel()
                    self._savePrediction()
        return self.model, self.lsTrainAccuracy, self.lsTestAccuracy

    def _train(self, epoch):
        correct, train_loss = 0, 0
        self.model.train()
        if self.lr_decay:
            if (self.lr_decay[4] == 'linear'):
                lr = max(self.lr_decay[0] * (1 - self.lr_decay[1] * (epoch // self.lr_decay[2]) ), self.lr_decay[3]) # Linear decay
            elif (self.lr_decay[4] == 'exp'):
                lr = max(self.lr_decay[0] * (self.lr_decay[1] ** (epoch // self.lr_decay[2])), self.lr_decay[3]) # Exponential decay
            if self.lr_decay[4] != 'None':
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        j, nRec = 0, 0

        self.Y_train = []
        self.target_train = []

        while j <= self.n_batch: # Use this to adjust how many batches as one epoch
            # for batch_idx, sample in enumerate(self.train_loader):
            sample = self.train_loader.__iter__().__next__()
            set1, set2, len1, len2, labels, genre = sample['set1'], sample['set2'], sample['len1'], \
                                             sample['len2'], sample['labels'], sample['genre']

            nRec += set1.size()[0]
            set1, set2, labels, len1, len2 = Variable(set1).long(), Variable(set2).long(), Variable(labels).long(), Variable(len1).long(), Variable(len2).long()

            self.cnt_iter += 1

            if self.flg_cuda:
                set1, set2, labels, len1, len2 = set1.cuda(), set2.cuda(), labels.cuda(), len1.cuda(), len2.cuda()

            self.optimizer.zero_grad()
            output = self.model(set1, set2, len1, len2)
            loss = self.criterion(output, labels)

            #if self.alpha_L1 > 0:
            #    l1_crit = nn.L1Loss(size_average=False)
            #    for fc in self.model.FCs:
            #        if self.flg_cuda:
            #            target_reg = Variable(torch.zeros(fc.weight.size())).cuda()
            #        else:
            #            target_reg = Variable(torch.zeros(fc.weight.size()))
            #        loss += l1_crit(fc.weight, target_reg) * self.alpha_L1

            loss.backward()
            self.optimizer.step()

            self.Y_train.append(output.data.cpu().numpy())
            self.target_train.append(labels.data.cpu().numpy())

            correct += self._getAccuracy(output, labels)
            train_loss += loss.data.item()
            j += 1
            if (j % self.log_interval[1] == 0):
                train_loss_temp = train_loss / nRec
                print('Train Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch, j, train_loss_temp))

        if (epoch == 0) | (epoch % self.log_interval[0] == 0) | (epoch == self.n_iter - 1):
            trainAccuracy = 100. * correct / nRec
            train_loss /= nRec
            self.lsTrainAccuracy.append(trainAccuracy)
            print('\nTrain Epoch: {} Loss: {:.4f} Accuracy: {:.4f}'.format(epoch, train_loss, trainAccuracy))


    def _test(self, epoch):
        if (epoch == 0) | (epoch % self.log_interval[0] == 0) | (epoch == self.n_iter - 1):

            self.model.eval()
            test_loss = 0
            correct, nRec = 0, 0

            self.Y = []
            self.target = []
            self.genre = []

            for batch_idx, sample in enumerate(self.test_loader):
                set1, set2, len1, len2, labels, genre = sample['set1'], sample['set2'], sample['len1'], \
                                                        sample['len2'], sample['labels'], sample['genre']


                nRec += set1.size()[0]

                set1, set2, labels, len1, len2 = Variable(set1).long(), Variable(set2).long(), Variable(
                    labels).long(), Variable(len1).long(), Variable(len2).long()

                if self.flg_cuda:
                    set1, set2, labels, len1, len2 = set1.cuda(), set2.cuda(), labels.cuda(), len1.cuda(), len2.cuda()

                with torch.no_grad():
                    output = self.model(set1, set2, len1, len2)
                    test_loss += (self.criterion(output, labels)).data.item()
                    correct += self._getAccuracy(output, labels)

                self.Y.append(output.data.cpu().numpy())
                self.target.append(labels.data.cpu().numpy())
                self.genre.extend(genre)

            testAccuracy = 100. * correct / nRec
            test_loss /= nRec  # loss function already averages over batch size
            print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, testAccuracy))
            self.lsTestAccuracy.append(testAccuracy)
            self.lsEpochNumber.append(epoch)
            self.acc = correct / nRec

    def _getAccuracy(self, output, target):
        pred = output.data.argmax(dim = 1)
        accuracy = pred.eq(target.data).cpu().float().numpy()
        accuracy = np.sum(accuracy)
        return accuracy

    def _saveModel(self):
        torch.save(self.model, self.savePath + '_model.pt')

    def _savePrediction(self, saveName=''):
        Y_hat = np.concatenate(self.Y)
        Y_hat_class = np.argmax(Y_hat.data, axis = 1)
        Y = np.concatenate(self.target)
        pickle.dump([Y_hat, Y_hat_class, Y, self.genre], open(self.savePath + str(saveName) + '_pred.p', 'wb'))
