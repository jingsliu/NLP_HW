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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--modelName", default="BagOfWords")
    parser.add_argument("--doc_len", type=int, default=300)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay3", type=int, default=1)  # Decay learning rate every lr_decay3 epochs
    parser.add_argument("--lr_decay_type", default='linear')  # Decay learning rate by linear or exp

    parser.add_argument("--i", type=int, default=1)  # Index of the element in the parameter set to be tuned
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--flg_cuda", action='store_true')

    parser.add_argument("--dimEmb", type=int, default=100)  # Dimension of embedding
    parser.add_argument("--nVocab", type=int, default=20000)  # Vocabulary size
    parser.add_argument("--optType", default='Adam')  # Vocabulary size

    parser.add_argument("--logInterval", type=int, default=1)  # Print test accuracy every n epochs
    parser.add_argument("--flgSave", action='store_true')
    parser.add_argument("--savePath", default='./')
    parser.add_argument("--randSeed", type=int, default=42)
    parser.add_argument("--inputPath", default="../aclImdb/df07f20K_stopEng_W_1gram/")

    args = parser.parse_args()

    torch.manual_seed(args.randSeed)  # For reproducible results
    if args.flgSave:
        if not os.path.isdir(args.savePath):
            os.mkdir(args.savePath)

    print('General parameters: ', args)

    print("Loading Data")
    # if args.modelName in ['Enc_SumLSTM', 'Enc_CNN_LSTM']:
    trainset = m.MovieDataset(args.inputPath, 'train.json', transform=m.padToTensor(args.doc_len))
    testset = m.MovieDataset(args.inputPath, 'val.json', transform=m.padToTensor(args.doc_len))

    print('To Loader')
    if args.flg_cuda:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, pin_memory=True)
    else:
        train_loader_pos = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, pin_memory=False)

    model_paras = {'doc_len': args.doc_len, 'dimEmb': args.dimEmb, 'nVocab': args.nVocab}

    print('Model parameters: ', model_paras)

    model = getattr(m, args.modelName)(model_paras)

    if args.flg_cuda:
        model = model.cuda()

    print(model)

    if args.optType == 'Adam':
        opt = optim.Adam(model.params, lr=args.lr)
    elif args.optType == 'SGD':
        opt = optim.SGD(model.params, lr=args.lr)

    print("Beginning Training")
    train_paras = {'n_iter': args.n_iter, 'log_interval': [args.logInterval, 1000], 'flg_cuda': args.flg_cuda,
                   'lr_decay': [args.lr, 0.9, args.lr_decay3, 1e-5, args.lr_decay_type],
                   'flgSave': args.flgSave, 'savePath': args.savePath}

    m = m.trainModel(train_paras, train_loader, test_loader, model, opt)
    start = time.time()
    _, lsTrainAccuracy, lsTestAccuracy = m.run()
    print('Test F1 max: %.3f' % (np.max(lsTestAccuracy)))
    print('Test F1 final: %.3f' % (lsTestAccuracy[-1]))
    stopIdx = min(lsTestAccuracy.index(np.max(lsTestAccuracy)) * args.logInterval, args.n_iter)
    print('Stop at: %d' % (stopIdx))
    end = time.time()
    print('Training time: ', end - start)


