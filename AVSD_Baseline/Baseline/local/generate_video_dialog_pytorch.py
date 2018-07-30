#!/usr/bin/env python
"""Video Caption Generation
   Copyright 2016 Mitsubishi Electric Research Labs
"""

import argparse
import math
import sys
import time
import os
import copy
import pickle

import numpy as np
import six
import torch
import torch.nn as nn
import pickle
import data_handler as dh



# Evaluation routine
def generate_caption(model, data, batch_indices, vocab, dim, stride=1, maxlen=20, beam=5, penalty=2.0, nbest=1):
    vocablist = sorted(vocab.keys(), key=lambda s: vocab[s])
    result = []
    c = 0
    for j in six.moves.range(len(batch_indices[0])):
        start_time = time.time()
        x_batch = [None] * len(data)
        for n in six.moves.range(len(data)):
            x_batch[n], Q_batch, A_batch = dh.make_batch(data[n], batch_indices[n][j],
                                                dim=dim[n], stride=stride)


        pred_out, logp = model.generate(x_batch, Q_batch, A_batch, maxlen=maxlen, beam=beam, penalty=penalty)

        for i in six.moves.range(Q_batch.shape[1]):
            c=c+1
            print c, batch_indices[0][j][0][0]+'_'+str(i+1)
            print 'REF:',
            for n in six.moves.range(A_batch.shape[0]):
                number = A_batch[n][i][0]
                if number == 3:
                    continue
                else:
                    print vocablist[A_batch[n][i][0]],
            print
            for n in six.moves.range(min(nbest, len(pred_out[i]))):
                # pdb.set_trace()
                print 'HYP[%d]:' % (n + 1),
                pred = pred_out[i][n]
                if (isinstance(pred, list)):
                    # The format used in run_youtube_mm.sh.
                    for w in pred:
                        print vocablist[w],
                    print '( {:f} )'.format(logp[i][n]);
                else:
                    # The format used in run_youtube.sh.
                    print(vocablist[pred_out[i][n]]);
                    print('( {:f} )'.format(logp[i]));
            print 'ElapsedTime:', time.time() - start_time
            print '-----------------------'
            ## print sentence
        result.append(pred_out)
    return result


##################################
# main
if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--capfile', default='', type=str,
                        help='Caption file (.json)')
    parser.add_argument('--feafile', nargs='+', type=str,
                        help='Image feature file (.pkl)')
    parser.add_argument('--test', default='', type=str,
                        help='Filename of test data')
    parser.add_argument('--model', '-m', default='', type=str,
                        help='Attention model to be output')
    parser.add_argument('--maxlen', default=20, type=int,
                        help='Max-length of output sequence')
    parser.add_argument('--beam', default=3, type=int,
                        help='Beam width')
    parser.add_argument('--penalty', default=2.0, type=float,
                        help='Insertion penalty')
    parser.add_argument('--nbest', default=5, type=int,
                        help='Number of n-best hypotheses')

    args = parser.parse_args()
    print 'Loading model params from', args.model
    path = args.model+'_parameter'
    with open(path, 'r') as f:
        vocab, train_args = pickle.load(f)
    model = torch.load(args.model+'.pth.tar')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)


    # prepare test data
    print 'Making mini batches for test data from', args.test
    data = [];

    for n, feafile in enumerate(args.feafile):
        feature_data = dh.load(feafile, args.capfile, vocab=vocab)
        data.extend(dh.check_feature_shape(feature_data));

    test_indices = [None] * len(args.feafile)

    for n,feafile in enumerate(args.feafile):
        test_indices[n], test_samples = dh.make_batch_indices(data[n], args.test, 1, test=True)
        print 'Feature[%d]: #test sample = %d  #test batch = %d' % (n, test_samples, len(test_indices[n]))

    print '#vocab =', len(vocab)
    # generate sentences
    print '-----------------------generate--------------------------'
    start_time = time.time()
    result = generate_caption(model, data, test_indices, vocab, train_args.in_size,  stride=train_args.frame_stride, maxlen=args.maxlen, beam=args.beam, penalty=args.penalty, nbest=args.nbest)
    print '----------------'
    print 'wall time =', time.time() - start_time
    print 'done'

