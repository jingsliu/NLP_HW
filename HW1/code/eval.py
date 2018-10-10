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
    parser.add_argument("--doc_len", type=int, default=300)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--flg_cuda", action='store_true')

    parser.add_argument("--optType", default='Adam')  # Vocabulary size

    parser.add_argument("--logInterval", type=int, default=1)  # Print test accuracy every n epochs
    parser.add_argument("--flgSave", action='store_true')
    parser.add_argument("--savePath", default='./')
    parser.add_argument("--randSeed", type=int, default=42)
    parser.add_argument("--inputPath", default="../aclImdb/df07f20K_stopEng_W_1gram/")
    parser.add_argument("--modelPath")

    args = parser.parse_args()

    torch.manual_seed(args.randSeed)  # For reproducible results

    if not os.path.isdir(args.savePath):
        os.mkdir(args.savePath)

    print('General parameters: ', args)

    print("Loading Data")
    # if args.modelName in ['Enc_SumLSTM', 'Enc_CNN_LSTM']:
    #trainset = m.MovieDataset(args.inputPath, 'train.json', transform=m.padToTensor(args.doc_len))
    testset = m.MovieDataset(args.inputPath, 'test.json', transform=m.padToTensor(args.doc_len))

    print('To Loader')
    if args.flg_cuda:
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, pin_memory=True)
    else:
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, pin_memory=False)

    print("Loading model")
    if args.flg_cuda:
        model = torch.load(args.modelPath + '_model.pt')
        model = model.cuda()
    else:
        model = torch.load(args.modelPath + '_model.pt', map_location=lambda storage, loc: storage)

    print(model)

    if args.optType == 'Adam':
        opt = optim.Adam(model.params, lr=args.lr)
    elif args.optType == 'SGD':
        opt = optim.SGD(model.params, lr=args.lr)

    print("Beginning Training")
    train_paras = {'log_interval': [args.logInterval, 1000], 'flg_cuda': args.flg_cuda,
                   'flgSave': args.flgSave, 'savePath': args.savePath}

    m = m.trainModel(train_paras, None, test_loader, model, opt)
    m._test(0)


