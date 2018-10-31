# Train model
import sys
import time
import numpy as np
import torch.optim as optim
import pickle
import os
import torch.utils.data
import model as m
import argparse
import pdb

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--modelName", default="CNN")
    parser.add_argument("--doc_len", type=int, default=20)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay_rate", type=float, default=0.9) # Rate of learning rate decay
    parser.add_argument("--lr_decay3", type=int, default=5)  # Decay learning rate every lr_decay3 epochs
    parser.add_argument("--lr_decay_type", default='exp')  # Decay learning rate by linear or exp

    parser.add_argument("--i", type=int, default=1)  # Index of the element in the parameter set to be tuned
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--flg_cuda", action='store_true')
    parser.add_argument("--n_batch", type=int, default=1)

    #parser.add_argument("--dimEmb", type=int, default=300)  # Dimension of embedding
    #parser.add_argument("--nVocab", type=int, default=20000)  # Vocabulary size
    parser.add_argument("--optType", default='Adam')  # optimizer
    parser.add_argument("--filters1", type=int, default=64)
    parser.add_argument("--filters2", type=int, default=64)
    parser.add_argument("--K1", type=int, default=5)
    parser.add_argument("--K2", type=int, default=3)
    parser.add_argument("--L1", type=int, default=32)
    parser.add_argument("--p_dropOut", type=float, default=0.5)
    parser.add_argument("--flgProd", action='store_true') # Whether to take element-wise product of the encoder outputs or concat
    parser.add_argument("--dimLSTM", type=int, default=128)

    parser.add_argument("--logInterval", type=int, default=1)  # Print test accuracy every n epochs
    parser.add_argument("--flgSave", action='store_true')
    parser.add_argument("--savePath", default='./')
    parser.add_argument("--randSeed", type=int, default=42)
    parser.add_argument("--inputPath", default="../hw2_data/df07f20K/")

    args = parser.parse_args()

    """
    args.filters1, args.filters2, args.L1 = [[512, 256, 128], [256, 256, 128], [256, 128, 64],
                                             [128, 64, 64], [64, 64, 32],[64, 32, 16]][args.i - 1]
    
    args.K1, args.K2 = [[2,2], [3,2], [3,3], [3,5], [4,2], [4,3], [4,4],
                        [5,3], [5,5], [5,7], [7,3], [7,5], [7,7], [9,3],
                        [9,5]][args.i - 1]
    """
    args.dimLSTM, args.L1 = [[256, 512],[256, 256],[128, 256], [128, 128],
                             [64, 128], [64, 64], [32,64], [32, 32]][args.i - 1]

    torch.manual_seed(args.randSeed)  # For reproducible results

    if not os.path.isdir(args.savePath):
        os.mkdir(args.savePath)
        pickle.dump(args, open(args.savePath + '_params.p', 'wb'))

    print('General parameters: ', args)

    print("Loading Data")
    trainset = m.setDataset(args.inputPath, 'train.p', transform=m.padToTensor(args.doc_len))
    testset = m.setDataset(args.inputPath, 'test.p', transform=m.padToTensor(args.doc_len))
    emb = pickle.load(open(args.inputPath + 'embedding.p', 'rb'))
    nVocab, dimEmb = emb.shape
    emb = torch.from_numpy(emb).float()

    print('To Loader')
    if args.flg_cuda:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, pin_memory=False)

    model_paras = {'doc_len': args.doc_len, 'dimEmb': dimEmb, 'nVocab': nVocab, 'p_dropOut': args.p_dropOut, 'flg_bn': True,
                   'filters1': args.filters1, 'filters2': args.filters2, 'K1': args.K1, 'K2': args.K2, 'L1': args.L1,
                   'flgProd': args.flgProd, 'dimLSTM': args.dimLSTM, 'flg_cuda': args.flg_cuda
                   }

    print('Model parameters: ', model_paras)

    model = getattr(m, args.modelName)(model_paras, emb)

    if args.flg_cuda:
        model = model.cuda()

    print(model)

    if args.optType == 'Adam':
        opt = optim.Adam(model.params, lr=args.lr)
    elif args.optType == 'SGD':
        opt = optim.SGD(model.params, lr=args.lr)

    print("Beginning Training")
    train_paras = {'n_iter': args.n_iter, 'log_interval': [args.logInterval, 1000], 'flg_cuda': args.flg_cuda,
                   'lr_decay': [args.lr, args.lr_decay_rate, args.lr_decay3, 1e-5, args.lr_decay_type],
                   'flgSave': args.flgSave, 'savePath': args.savePath, 'n_batch': args.n_batch}

    m = m.trainModel(train_paras, train_loader, test_loader, model, opt)
    start = time.time()
    _, lsTrainAccuracy, lsTestAccuracy = m.run()
    print('Test Acc max: %.3f' % (np.max(lsTestAccuracy)))
    print('Test Acc final: %.3f' % (lsTestAccuracy[-1]))
    stopIdx = min(lsTestAccuracy.index(np.max(lsTestAccuracy)) * args.logInterval, args.n_iter)
    print('Stop at: %d' % (stopIdx))
    end = time.time()
    print('Training time: ', end - start)


