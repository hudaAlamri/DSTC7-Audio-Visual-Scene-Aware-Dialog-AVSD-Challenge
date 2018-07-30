#!/usr/bin/env python
"""Train an encoder-decoder model for video captioning
   Copyright 2015-2016 Mitsubishi Electric Research Labs
"""

import argparse
import logging;
import math
import sys
import time
import random
import os
import copy
import importlib

import numpy as np
import six

import torch
import torch.nn as nn
import pickle
import data_handler as dh
from sklearn import preprocessing

def evaluate(model, data, batch_indices, dim, stride=1):
    start_time = time.time()
    eval_loss = 0.
    eval_hit = 0
    num_tokens = 0
    for j in six.moves.range(len(batch_indices)):
        x_batch = [None] * len(data)
        for m in six.moves.range(len(data)):
            x_batch[m], Q_batch, A_batch = dh.make_batch(data[m], batch_indices[m][j],
                                             dim=dim[m], stride=stride)
        loss, hit, num = model(x_batch, Q_batch, A_batch, predicted_context=False, istraining=False)
        eval_loss += loss
        eval_hit += hit
        num_tokens += num

    wall_time = time.time() - start_time
    return math.exp(eval_loss/num_tokens), float(eval_hit)/num_tokens, wall_time

##################################
# main
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')

    # train, dev and test data
    parser.add_argument('--vocabfile', default='', type=str, 
                        help='Vocabulary file (.json)')
    parser.add_argument('--capfile', default='', type=str, 
                        help='Caption file (.json)')
    parser.add_argument('--feafile', nargs='+', type=str, 
                        help='Image feature files (.pkl)')
    parser.add_argument('--train', default='train.list', type=str,
                        help='Filename of train data')
    parser.add_argument('--valid', default='dev.list', type=str,
                        help='Filename of validation data')
    parser.add_argument('--test', default='', type=str,
                        help='Filename of test data')
    
    # Attention model related
    parser.add_argument('--type', '-t', default='arsg', type=str, 
                        help="model type")
    parser.add_argument('--initial-model', '-i', default='', type=str,
                        help='Attention model to be used')
    parser.add_argument('--model', '-m', default='', type=str,
                        help='Attention model to be output')
    parser.add_argument('--num-epochs', '-e', default=20, type=int,
                        help='Number of epochs')
    parser.add_argument('--in-size', nargs='+', type=int,
                        help='Number of input units')
    parser.add_argument('--enc-hsize', '-u', nargs='+', type=int,
                        help='Number of hidden units')
    parser.add_argument('--enc-psize', '-p', nargs='+', type=int,
                        help='Number of projection layer units')
    parser.add_argument('--att-size', '-a', default=100, type=int,
                        help='Number of attention layer units')
    parser.add_argument('--dec-psize', '-P', default=200, type=int,
                        help='Number of decoder projection layer units')
    parser.add_argument('--dec-hsize', '-d', default=200, type=int,
                        help='Number of decoder hidden units')
    parser.add_argument('--predicted-context', action='store_true',
                        help='Use predicted context in training')

    # Training conditions
    parser.add_argument('--optimizer', '-o', default='AdaDelta', type=str, 
                        help="optimizer (SGD, Adam, AdaDelta)")
    parser.add_argument('--L2-weight', default=0.0005, type=float, 
                        help="Set weight of L2-regularization term")
    parser.add_argument('--clip-grads', default=5., type=float, 
                        help="Set gradient clipping threshold")
    parser.add_argument('--rand-seed', '-s', default=1, type=int, 
                        help="seed for generating random numbers")
    parser.add_argument('--learn-rate', '-R', default=0.1, type=float,
                        help='Initial learning rate')
    parser.add_argument('--learn-decay', '-D', default=0.01, type=float,
                        help='Decaying ratio of learning rate')
    parser.add_argument('--lower-bound', default=1e-16, type=float,
                        help='Lower-bound of learning rate or eps')
    parser.add_argument('--batch-size', '-b', default=20, type=int,
                        help='Batch size in training')
    parser.add_argument('--max-length', default=20, type=int,
                        help='Maximum length for controling batch size')
    parser.add_argument('--frame-stride', default=1, type=int,
                        help='Stride for input frame skipping')

    args = parser.parse_args()

    logging.warn("The argument in_size is ignored. The feature lengths will be decided from the data.");

    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    batchsize = args.batch_size  # number of words for truncation

    # load data
    vocab = {'<unk>': 0, '<sos>': 1, '<eos>': 2, '<no_tag>': 3}
    data = [];
    enc_hsizes = [];
    enc_psizes = [];

    for a in range(0,1):
        feafile = args.feafile
        print ('Loading data from', feafile, 'and', args.capfile)

    for n, feafile in enumerate(args.feafile):
        print('Loading data from ', feafile, ' and ', args.capfile)
        enc_hsize = args.enc_hsize[n];
        enc_psize = args.enc_psize[n];
        feature_data = dh.load(feafile, args.capfile, vocabfile=args.vocabfile, vocab=vocab);
        feature_data = dh.check_feature_shape(feature_data);
        n_features = len(feature_data);
        data.extend(feature_data);
        enc_hsizes.extend([enc_hsize] * n_features);
        enc_psizes.extend([enc_psize] * n_features);

    data_sizes = map(lambda d: d["feature"].values()[0].shape[-1], data);

    logging.warn("The detectected feature lengths are: {}".format(data_sizes));

    # A workaroung to reduce the lines to change later.
    args.in_size = data_sizes;
    args.enc_hsize = enc_hsizes;
    args.enc_psize = enc_psizes;

    # Prepare RNN model and load data
    if args.initial_model != '':
        print ('Loading model params from', args.initial_model)
        with open(args.initial_model, 'rb') as f:
            vocab, model, tmp_args = pickle.load(f)
    else:
        # make a model instance
        if 'arsgmm' in args.type:
            arsgmm = importlib.import_module(args.type)
            model = arsgmm.ARSGMM(args.in_size, len(vocab),
                        enc_psize=args.enc_psize,
                        enc_hsize=args.enc_hsize,
                        dec_psize=args.dec_psize,
                        dec_hsize=args.dec_hsize,
                        att_size=args.att_size,
                        sos=1, eos=2, ignore_label=3)
        else:
            print("Unknonw model type '{}' is specified.".format(args.type));
            sys.exit(1);

    # report data summary
    print ('#vocab =', len(vocab))
    
    # make batchset for training
    print ('Making mini batches for training data')
    train_indices = [None] * len(data)
    train_indices[0], train_samples = dh.make_batch_indices(data[0], args.train, 
                                    batchsize, max_length=args.max_length)
    for n in six.moves.range(1,len(data)):
        train_indices[n], _ = dh.make_batch_indices(data[n], args.train, batchsize, 
                         max_length=args.max_length, reference=train_indices[0])
    print ('#train sample =', train_samples, ' #train batch =', len(train_indices[0]) )

    if args.valid != '':
        print ('Making mini batches for validation data')
        valid_indices = [None] * len(data)
        valid_indices[0], valid_samples = dh.make_batch_indices(data[0], args.valid,
                                         batchsize, max_length=args.max_length)
        for n in six.moves.range(1,len(data)):
            valid_indices[n], _ = dh.make_batch_indices(data[n], args.valid, 
                                         batchsize, max_length=args.max_length,
                                         reference=valid_indices[0])
        print ('#valid sample =', valid_samples, ' #valid batch =', len(valid_indices[0]))

    if args.test != '':
        print ('Making mini batches for test data')
        test_indices = [None] * len(data)
        test_indices[0], test_samples = dh.make_batch_indices(data[0], args.test,
                                         batchsize, max_length=args.max_length)
        for n in six.moves.range(1,len(data)):
            test_indices[n], _ = dh.make_batch_indices(data[n], args.test, 
                                         batchsize, max_length=args.max_length,
                                         reference=test_indices[0])
        print ('#test sample =', test_samples, '#test batch =', len(test_indices[0]))

    # write initial model
    print ('----------------')
    print ('Writing initial model params to', args.model + '_0.pth.tar')
    torch.save(model, args.model + '_0.pth.tar')


    # start training 
    print ('----------------')
    print ('Start training')
    print ('----------------')
    # move model to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpu = torch.cuda.device_count()
    if num_gpu > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        multiGPU=True
    else:
        multiGPU=False
    model.to(device)
    # model=model.cuda()
    # Setup optimizer
    is_sgd = False
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=args.learn_rate,momentum=args.momentum, weight_decay=args.weight_decay)
        is_sgd = True
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif args.optimizer == 'AdaDelta':
        optimizer = torch.optim.Adadelta(model.parameters())
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters())
    else:
        print ('Unknown optimizer:', args.optimizer)
        sys.exit(1)
    # model.train()
    # initialize status parameters
    cur_log_perp = 0.
    num_words = 0
    epoch = 0
    start_at = time.time()
    cur_at = start_at
    max_valid_acc = 0.
    n = 0
    report_interval = 1000/batchsize
    bestmodel_num = 0
    # prepare data ids
    ids = range(len(train_indices[0]))
    random.shuffle(ids)
    # do training iterations
    for i in six.moves.range(args.num_epochs):
        if is_sgd==True:
            print ('Epoch %d : SGD learning rate = %g' % (i+1, optimizer.lr))
        else:
            print ('Epoch %d : %s' % (i+1, args.optimizer))

        train_loss = 0.
        for j in six.moves.range(len(ids)):
            # prepare input data
            k = ids[j]
            x_batch = [None] * len(data)
            for m in six.moves.range(len(data)):
                x_batch[m], Q_batch, A_batch = dh.make_batch(data[m], train_indices[m][k],
                                        dim=args.in_size[m], stride=args.frame_stride)
            # propagate for training
            loss = model(x_batch, Q_batch, A_batch, predicted_context=False, istraining=True)
            if multiGPU:
                loss = loss.sum()
            wj = loss.cpu()
            cur_log_perp += wj.data.numpy()
            num_words += A_batch.shape[0]*A_batch.shape[1]
            if (n + 1) % report_interval == 0:
                now = time.time()
                throuput = report_interval / (now - cur_at)
                perp = math.exp(cur_log_perp / num_words)
                print(' iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(n + 1, perp, throuput))

                cur_at = now
                cur_log_perp = 0.
                num_words = 0
            n += 1

            # Run truncated BPTT
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ("epoch:", i+1, "train loss:", train_loss)

        # validation step 
        print ('-----------------------evaluate--------------------------')
        now = time.time()
        # model.eval()
        if args.valid != '':
            valid_ppl,valid_acc,valid_time = evaluate(model, data, valid_indices, dim=args.in_size, stride=args.frame_stride)
            print (' valid perplexity: %.4f  valid accuracy: %.4f %%' % (valid_ppl, valid_acc * 100.0))
        if args.test != '':
            test_ppl,test_acc,test_time = evaluate(model, data, test_indices, dim=args.in_size, stride=args.frame_stride)
            print (' test perplexity: %.4f  test accuracy: %.4f %%' % (test_ppl, test_acc * 100.0))
        # update the model via comparing with the highest accuracy
        modelfile = args.model + '_' + str(i+1) + '.pth.tar'
        print(' writing model params to', modelfile)
        torch.save(model, modelfile)

        if max_valid_acc < valid_acc:
            bestmodel_num = i+1
            print (' (valid accuracy improved %.4f -> %.4f)' % (max_valid_acc, valid_acc))
            max_valid_acc = valid_acc

        cur_at += time.time() - now  # skip time of evaluation and file I/O
        print ('----------------')

    # make a symlink to the best model
    print('the best model is epoch %d.' % bestmodel_num)
    print ('a symbolic link is made as',  args.model+'_best')
    if os.path.exists(args.model+'_best.pth.tar'):
        os.remove(args.model+'_best.pth.tar')
    os.symlink(os.path.basename(args.model+'_'+str(bestmodel_num)+'.pth.tar'),
               args.model+'_best.pth.tar')
    path = args.model+'_best_parameter'
    with open(path, 'wb') as f:
        pickle.dump((vocab, args), f, -1)
    print ('done')
