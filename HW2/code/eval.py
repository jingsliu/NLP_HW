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

    parser = argparse.ArgumentParser()

    parser.add_argument("--modelName", default="RNN")
    parser.add_argument("--doc_len", type=int, default=30)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay_rate", type=float, default=0.9) # Rate of learning rate decay
    parser.add_argument("--lr_decay3", type=int, default=5)  # Decay learning rate every lr_decay3 epochs
    parser.add_argument("--lr_decay_type", default='exp')  # Decay learning rate by linear or exp

    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--flg_cuda", action='store_true')
    parser.add_argument("--n_batch", type=int, default=1)

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

    parser.add_argument("--modelPath")

    args = parser.parse_args()

    torch.manual_seed(args.randSeed)  # For reproducible results

    if not os.path.isdir(args.savePath):
        os.mkdir(args.savePath)

    print('General parameters: ', args)

    print("Loading Data")
    # if args.modelName in ['Enc_SumLSTM', 'Enc_CNN_LSTM']:
    testset = m.setDataset(args.inputPath, 'test_mnli.p', transform=m.padToTensor(args.doc_len))
    emb = pickle.load(open(args.inputPath + 'embedding.p', 'rb'))
    nVocab, dimEmb = emb.shape
    emb = torch.from_numpy(emb).float()

    print('To Loader')
    if args.flg_cuda:
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, pin_memory=True)
    else:
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, pin_memory=False)

    model_paras = {'doc_len': args.doc_len, 'dimEmb': dimEmb, 'nVocab': nVocab, 'p_dropOut': args.p_dropOut,
                   'flg_bn': True,
                   'filters1': args.filters1, 'filters2': args.filters2, 'K1': args.K1, 'K2': args.K2, 'L1': args.L1,
                   'flgProd': args.flgProd, 'dimLSTM': args.dimLSTM, 'flg_cuda': args.flg_cuda
                   }

    print("Loading model")
    if args.flg_cuda:
        model0 = torch.load(args.modelPath + '_model.pt')
        model0 = model0.cuda()
    else:
        model0 = torch.load(args.modelPath + '_model.pt', map_location=lambda storage, loc: storage)

    model = getattr(m, args.modelName)(model_paras, emb)
    model.load_state_dict(model0.state_dict())

    print(model)

    opt = optim.Adam(model.params, lr=args.lr)

    train_paras = {'n_iter': args.n_iter, 'log_interval': [args.logInterval, 1000], 'flg_cuda': args.flg_cuda,
                   'lr_decay': [args.lr, args.lr_decay_rate, args.lr_decay3, 1e-5, args.lr_decay_type],
                   'flgSave': args.flgSave, 'savePath': args.savePath, 'n_batch': args.n_batch}

    m = m.trainModel(train_paras, None, test_loader, model, opt)
    m._test(0)
    m._savePrediction()


