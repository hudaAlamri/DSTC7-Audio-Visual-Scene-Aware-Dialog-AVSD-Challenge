#!/usr/bin/env python
"""Functions for feature data handling
   Copyright 2016 Mitsubishi Electric Research Labs
"""

import copy;
import logging;
import sys
import time
import os
import six
import pickle
import json
import numpy as np
from sklearn.preprocessing import normalize
import pdb

# Load text data
def load(feafile, capfile, vocabfile='', vocab={}):
    cap_data = json.load(open(capfile,'r'))
    if vocabfile != '':
        vocab_from_file = json.load(open(vocabfile,'r'))
        for w in vocab_from_file:
            if w not in vocab:
                vocab[w] = len(vocab)
    unk = vocab['<unk>']
    eos = vocab['<eos>']
    seq_info = {}
    for k,v in cap_data.items():
        seq=[]
        for e in v:
            qa = {}
            answer = e['answer'].split()
            question = e['question'].split()
            A_sentence = np.ndarray(len(answer)+1, dtype=np.int32)
            Q_sentence = np.ndarray(len(question) + 1, dtype=np.int32)
            for i,w in enumerate(answer):
                if w in vocab:
                    A_sentence[i] = vocab[w]
                else:
                    A_sentence[i] = unk
            for i,w in enumerate(question):
                if w in vocab:
                    Q_sentence[i] = vocab[w]
                else:
                    Q_sentence[i] = unk

            A_sentence[len(answer)] = eos
            Q_sentence[len(question)] = eos
            qa['qa_id'] = e['qa_id']
            qa['image_id'] = k
            qa['answer'] = A_sentence
            qa['question'] = Q_sentence
            seq.append(qa)
        seq_info[k] = seq

    data = {}
    data['QA'] = cap_data
    data['QA_code'] = seq_info
    data['vocab'] = vocab 
    feat = pickle.load(open(feafile,'r'))
            
    if feafile.find('frcnn') > -1: # means its the frcnn feature.
        data_mean = 0;    
        for vid in feat.keys():
            data_mean += np.mean(np.array(feat[vid][0]), axis=0)
    
        data_mean /= len(feat)        
        for vid in feat.keys():                    
            feat[vid] = normalize(np.array(feat[vid][0])-data_mean, axis=1, norm='l2')
            #feat[vid] = normalize(np.random.randn(feat[vid][0].shape[0], feat[vid][0].shape[1]), axis=1, norm='l2')
        
            #feat[vid] = np.array(feat[vid][0])
    else:                
        data_mean = 0;    
        for t in feat.keys():  
            data_mean += np.mean(feat[t],axis=0)
    
        data_mean /= len(feat)        
        wj=feat.keys()
        for t in feat.keys():
            feat[t] = normalize(feat[t]-data_mean, axis=1, norm='l2')        
            #feat[t] = feat[t]
        
    data['feature'] = feat
        
    return data 

# Setup mini-batches
def make_batch_indices(data, idfile, batchsize=100, max_length=20, test=False, reference=[]):
    # make mini batches
    if len(reference) == 0:
        idxlist = []
        for idx in json.load(open(idfile,'r')):
            if idx in data['feature']:
                x_len = len(data['feature'][idx])
                A_len = 0
                Q_len = 0
                for qa_pair in data['QA_code'][idx]:
                    A_length = len(qa_pair['answer'])
                    Q_length = len(qa_pair['question'])
                    A_len = max(A_length, A_len)
                    Q_len = max(Q_length, Q_len)
                if x_len>0:
                    idxlist.append((idx, x_len, Q_len, A_len))
    
        if batchsize>1:
            idxlist = sorted(idxlist, key=lambda s:-s[1])
    
        n_samples = len(idxlist)
        batch_indices = []
        bs = 0
        while bs < n_samples:
            x_len = idxlist[bs][1]
            bsize = batchsize / (x_len / max_length + 1)
            be = min(bs + bsize, n_samples) if bsize > 0 else bs + 1
            Q_len = max(idxlist[bs:be], key=lambda s:s[2])[2]
            A_len = max(idxlist[bs:be], key=lambda s:s[3])[3]
            vids = [ s[0] for s in idxlist[bs:be] ]
            batch_indices.append((vids, x_len, Q_len,A_len, be - bs))
            bs = be
    else:
        batch_indices = []
        n_samples = 0
        for vids, x_len, Q_len, A_len, bsize in reference:
            x_len = max([len(data['feature'][vid]) for vid in vids])
            # y_len = max([len(data['sentence'][idx]) for idx in cids])
            batch_indices.append((vids, x_len, Q_len, A_len, bsize))
            n_samples += bsize
            
    return batch_indices, n_samples


def make_batch(data, index, dim=1024, no_tag=3, stride=1, num_qa=10):
    x_len = index[1]/stride if index[1] % stride == 0 else index[1]/stride + 1
    Q_len = index[2]
    A_len = index[3]
    n_seqs = index[4]
    x_data = data['feature']
    qa_code = data['QA_code']
    x_batch = np.zeros((x_len, n_seqs, dim), dtype=np.float32)
    Q_batch = np.ndarray((Q_len, num_qa, n_seqs), dtype=np.int32)
    A_batch = np.ndarray((A_len, num_qa, n_seqs), dtype=np.int32)
    Q_batch.fill(no_tag)
    A_batch.fill(no_tag)
    for i in six.moves.range(n_seqs):
        if index[0][i] in x_data:
            fea = x_data[index[0][i]]
            k = 0
            for j in six.moves.range(0,fea.shape[0],stride):
                x_batch[k][i] = fea[j]
                k += 1
            QA = qa_code[index[0][i]]
            for h in six.moves.range(num_qa):
                qa_pair = QA[h]
                answer = qa_pair['answer']
                question = qa_pair['question']
                a_len = len(answer)
                q_len = len(question)
                Q_batch[:q_len,h,i] = question
                A_batch[:a_len,h,i] = answer
    return x_batch, Q_batch, A_batch


def check_feature_shape(feature_data, w_unwarpping_features = True):
    w_spatial_dims = False;

    data = [];
    # Get one sample to check the shape.
    image_features = feature_data["feature"];
    sample_feature = image_features.values()[0];
    if (3 <= len(sample_feature.shape)):
        w_spatial_dims = True;

    if( w_spatial_dims ):
        def split_features():
            new_feature_data = None;
            for image_id, image_feature in image_features.iteritems():
                reshaped_feature = np.reshape(image_feature, list(image_feature.shape[:2]) + [-1]);
                n_features = reshaped_feature.shape[-1];
                if( None is new_feature_data ):
                    new_feature_data = [];
                    for _ in xrange(n_features):
                        new_feature_data.append(copy.copy(feature_data));
                for fi in xrange(n_features):
                    new_feature_data[fi]["feature"][image_id] = np.squeeze(reshaped_feature[..., fi]);

            data.extend(new_feature_data);

        def unwarp_features():
            new_feature_data = copy.copy(feature_data);
            for image_id, image_feature in image_features.iteritems():
                reshaped_feature = np.reshape(image_feature, [image_feature.shape[0], -1]);
                new_feature_data["feature"][image_id] = reshaped_feature;
            data.append(new_feature_data);

        # For now only allow to unwarp the features into vectors.
        if( w_unwarpping_features ):
            unwarp_features();
        else:
            split_features();
    else:
        data.append(feature_data);

    return data;
