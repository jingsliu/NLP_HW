

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics
import json
import pickle


#==== Data ======

class MovieDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, root_dir, dsName, transform=None):
        self.root_dir = root_dir
        self.ds = json.load(open(root_dir + dsName, 'r'))
        print('Loaded: ', dsName)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Triggered when you call dataset[i]
        """
        words, labels, ID = self.ds[idx][0:]
        words = np.asarray(words, dtype='int')
        labels = np.asarray(labels, dtype='int')

        sample = {'words': words, 'labels': labels, 'ID': ID}
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
        words, labels, ID = sample['words'], sample['labels'], sample['ID']
        return self._run(ID, words, labels)

    def _run(self, ID, words, labels):
        # Pad docs
        padded_words = self._pad_doc(words, self.doc_len)
        mask = torch.zeros(self.doc_len).float()
        mask[0:len(words)] = 1.0

        return {'ID': ID,
                'words': padded_words,
                'labels': torch.from_numpy(np.asarray(labels)).long(),
                'mask': mask
                }

    def _pad_doc(self, seq, max_len):
        padded_seq = torch.zeros(max_len)
        s = min(max_len, len(seq))
        padded_seq[0:s] = torch.from_numpy(np.asarray(seq[0:s])).long()
        return padded_seq

#===== Model ========

class BagOfWords(nn.Module):
    """
    BagOfWords classification model
    """

    def __init__(self, model_paras):
        """
        @param vocab_size: size of the vocabulary.
        @param emb_dim: size of the word embedding
        """
        super(BagOfWords, self).__init__()
        vocab_size = model_paras.get('nVocab', 20000)
        emb_dim = model_paras.get('dimEmb', 100)

        # pay attention to padding_idx
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim, 20)

    def forward(self, words, mask):
        """

        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        out = self.embed(words)
        out = torch.sum(out, dim=1)
        out /= torch.sum(mask, dim=1)

        # return logits
        out = self.linear(out.float())
        return out


#====== Training =======
class trainModel(object):
    def __init__(self, train_paras, train_loader, test_loader, model, optimizer):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer

        # self.train_paras = train_paras
        self.n_batch = train_paras.get('n_batch', 10) # Use this to adjust how many batches as one epoch
        self.n_iter = train_paras.get('n_iter', 1)
        self.log_interval = train_paras.get('log_interval', 1)
        self.flg_cuda = train_paras.get('flg_cuda', False)
        # self.max_len = train_paras.get('max_len', 2000) # Max length of input
        self.lr_decay = train_paras.get('lr_decay', None)  # List of 4 numbers: [init_lr, lr_decay_rate, lr_decay_interval, min_lr, decay_type]
        self.flgSave = train_paras.get('flgSave', False)  # Set to true if save model
        self.savePath = train_paras.get('savePath', './')
        self.alpha_L1 = train_paras.get('alpha_L1', 0.0)  # Regularization coefficient on fully connected weights

        if self.lr_decay:
            assert len(self.lr_decay) == 5  # Elements include: [starting_lr, decay_multiplier, decay_per_?_epoch, min_lr, decay_type]
        self.criterion = torch.nn.CrossEntropyLoss()
        self.cnt_iter = 0

        self.lsTrainAccuracy = []
        self.lsTestAccuracy = []
        self.lsEpochNumber = []
        self.bestAccuracy = 0.0
        self.auc = 0.0


    def run(self):
        for epoch in range(self.n_iter):
            self._train(epoch)
            self._test(epoch)
            pickle.dump([self.lsEpochNumber, self.lsTrainAccuracy, self.lsTestAccuracy], open(self.savePath + '_accuracy.p', 'wb'))
            if self.auc > self.bestAccuracy:
                self.bestAccuracy = self.auc
                if self.flgSave:
                    self._saveModel()
        return self.model, self.lsTrainAccuracy, self.lsTestAccuracy

    def _train(self, epoch, lsTrainAccuracy):
        correct, train_loss = 0, 0
        self.model.train()
        if self.lr_decay:
            if (self.lr_decay[4] == 'linear'):
                lr = min(self.lr_decay[0] / (epoch // self.lr_decay[2] + 1), self.lr_decay[3]) # Linear decay
            elif (self.lr_decay[4] == 'exp'):
                lr = min(self.lr_decay[0] * (self.lr_decay[1] ** (epoch // self.lr_decay[2])), self.lr_decay[3]) # Exponential decay
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        j, nRec = 0, 0

        self.Y_train = []
        self.target_train = []

        while j <= self.n_batch: # Use this to adjust how many batches as one epoch
            # for batch_idx, sample in enumerate(self.train_loader):
            sample = self.train_loader.__iter__().__next__()
            words, labels, mask = sample['words'], sample['labels'], sample['mask']

            nRec += words.size()[0]
            words, labels, mask = Variable(words).long(), Variable(labels).long(), Variable(mask).float()

            self.cnt_iter += 1

            if self.flg_cuda:
                words, labels, mask = words.cuda(), labels.cuda(), mask.cuda()

            self.optimizer.zero_grad()
            output = self.model(words, mask)
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
            train_loss += loss.data[0]
            j += 1
            if (j % self.log_interval[1] == 0):
                train_loss_temp = train_loss / np.sum(nRec)
                print('Train Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch, j, train_loss_temp))

        if (epoch == 0) | (epoch % self.log_interval[0] == 0) | (epoch == self.n_iter - 1):
            trainAccuracy = 100. * correct / nRec
            train_loss /= np.sum(nRec)
            self.lsTrainAccuracy.append(trainAccuracy)

            print('\nTrain Epoch: {} Loss: {:.4f} Accuracy: {:.4f}'.format(epoch, train_loss, trainAccuracy))


    def _test(self, epoch, lsTestAccuracy):
        if (epoch == 0) | (epoch % self.log_interval[0] == 0) | (epoch == self.n_iter - 1):

            self.model.eval()
            test_loss = 0
            correct, nRec = 0, 0

            self.Y = []
            self.target = []

            for batch_idx, sample in enumerate(self.test_loader):
                words, labels, mask = sample['words'], sample['labels'], sample['mask']
                nRec += words.size()[0]

                words, labels, mask = Variable(words, volatile=True).long(), Variable(labels,volatile=True).long(), Variable(mask, volatile=True).float()
                if self.flg_cuda:
                    words, labels, mask = words.cuda(), labels.cuda(), mask.cuda()
                with torch.no_grad():
                    output = self.model(words, labels, mask)
                    test_loss += (self.criterion(output, labels)).data[0]
                    correct += self._getAccuracy(output, labels)

                self.Y.append(output.data.cpu().numpy())
                self.target.append(labels.data.cpu().numpy())

            testAccuracy = 100. * correct / nRec
            test_loss /= np.sum(nRec)  # loss function already averages over batch size
            print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, testAccuracy))
            self.lsTestAccuracy.append(testAccuracy)
            self.lsEpochNumber.append(epoch)

    def _getAccuracy(self, output, target):
        pred = output.max(1, keepdim=True)[1]
        accuracy = pred.eq(target.data).cpu().float().numpy()
        accuracy = np.sum(accuracy, axis=0)
        return accuracy

    def _saveModel(self):
        torch.save(self.model, self.savePath + '_model.pt')
