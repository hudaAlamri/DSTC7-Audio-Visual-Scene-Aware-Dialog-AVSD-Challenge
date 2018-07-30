#!/usr/bin/env python
"""ARSGMM: Attention-based Recurrent Sequence Generator for Multi-modal Inputs 
   Copyright 2016 Mitsubishi Electric Research Labs
"""

import math
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


class ARSGMM(nn.Module):

    def __init__(self, in_size, out_size, 
                 enc_psize=[], enc_hsize=[], dec_psize=100, dec_hsize=100, att_size=100,
                 matt_size=0, sos=1, eos=2, ignore_label=3, inp = 1024, mid = 1024, sz = 3):
        if len(enc_psize)==0:
            enc_psize = in_size
        if len(enc_hsize)==0:
            enc_hsize = [0] * len(in_size)
        if matt_size==0:
            matt_size = att_size

        # make links
        super(ARSGMM, self).__init__()
        self.inp = inp
        self.mid = mid

        # memorize sizes
        self.n_inputs = len(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.enc_psize = enc_psize
        self.enc_hsize = enc_hsize
        self.att_size = att_size
        self.dec_psize = dec_psize
        self.dec_hsize = dec_hsize
        self.ignore_label = ignore_label
        self.sos = sos
        self.eos = eos

        # encoder
        self.l1f_x = nn.ModuleList()
        self.l1f_h = nn.ModuleList()
        self.l1b_x = nn.ModuleList()
        self.l1b_h = nn.ModuleList()
        self.emb_x = nn.ModuleList()
        for m in six.moves.range(len(in_size)):
            self.emb_x.append(nn.Linear(self.in_size[m], self.enc_psize[m]))
            if enc_hsize[m] > 0:
                self.l1f_x.append(nn.Linear(enc_psize[m], 4 * enc_hsize[m]))
                self.l1f_h.append(nn.Linear(enc_hsize[m], 4 * enc_hsize[m], bias = False))
                self.l1b_x.append(nn.Linear(enc_psize[m], 4 * enc_hsize[m]))
                self.l1b_h.append(nn.Linear(enc_hsize[m], 4 * enc_hsize[m], bias = False))

        # attention
        self.atV = nn.ModuleList()
        self.atW = nn.ModuleList()
        self.atW_qa = nn.ModuleList()
        self.atW_qq = nn.ModuleList()
        self.atw = nn.ModuleList()
        self.lgd = nn.ModuleList()

        for m in six.moves.range(len(in_size)):
            enc_hsize_ = 2*enc_hsize[m] if enc_hsize[m] > 0 else enc_psize[m]
            self.atV.append(nn.Linear(enc_hsize_, att_size))
            self.atW.append(nn.Linear(dec_hsize, att_size))
            self.atW_qa.append(nn.Linear(dec_hsize, att_size))
            self.atW_qq.append(nn.Linear(dec_hsize, att_size))
            self.atw.append(nn.Linear(att_size, 1))
            self.lgd.append(nn.Linear(enc_hsize_, dec_hsize))

        # decoder
        self.emb_y = nn.Embedding(out_size, dec_psize)
        self.l1d_y = nn.Linear(dec_psize, 4 * dec_hsize)
        self.l1d_y_q = nn.Linear(dec_psize, 4 * dec_hsize)
        self.l1d_y_a = nn.Linear(dec_psize, 4 * dec_hsize)
        self.l1d_y_qq = nn.Linear(dec_psize, 4 * dec_hsize)
        self.l1d_h = nn.Linear(dec_hsize, 4 * dec_hsize, bias=False)
        self.l1d_h_q = nn.Linear(dec_hsize, 4 * dec_hsize, bias=False)
	self.l1d_h_a = nn.Linear(dec_hsize, 4 * dec_hsize, bias=False)
        self.l1d_h_qq = nn.Linear(dec_hsize, 4 * dec_hsize, bias=False)
        self.l2d = nn.Linear(dec_hsize, dec_hsize)
        self.l4d = nn.Linear(dec_hsize, dec_hsize)
        self.l5d = nn.Linear(dec_hsize, dec_hsize)
        self.l3d = nn.Linear(4*dec_hsize, out_size)




    # Make an initial state
    def make_initial_state(self, hiddensize):
        return {name: torch.zeros(self.bsize, hiddensize, dtype=torch.float)
                    for name in ('c1', 'h1')}

    # Encoder functions
    def embed_x(self, x_data,m):
        x0 = [torch.from_numpy(x_data[i])
            for i in six.moves.range(len(x_data))]
        return self.emb_x[m](torch.cat(x0, 0).cuda().float())

    def forward_one_step(self, x, s, m):
        x_new = x + self.l1f_h[m](s['h1'].cuda())
        x_list = torch.split(x_new, self.enc_hsize[m], dim=1)
        x_list = list(x_list)
        c1 = F.tanh(x_list[0]) * F.sigmoid(x_list[1]) + s['c1'].cuda() * F.sigmoid(x_list[2])
        h1 = F.tanh(c1) * F.sigmoid(x_list[3])
        return {'c1': c1, 'h1': h1}

    def backward_one_step(self, x, s, m):
        x_new = x + self.l1b_h[m](s['h1'].cuda())
        x_list = torch.split(x_new, self.enc_hsize[m], dim=1)
        x_list = list(x_list)
        c1 = F.tanh(x_list[0]) * F.sigmoid(x_list[1]) + s['c1'].cuda() * F.sigmoid(x_list[2])
        h1 = F.tanh(c1) * F.sigmoid(x_list[3])
        return {'c1': c1, 'h1': h1}

    # Encoder main

    def encode(self, x):
        h1 = [None] * self.n_inputs
        for m in six.moves.range(self.n_inputs):
            # self.emb_x=self.__dict__['emb_x%d' % m]
            if self.enc_hsize[m] > 0:
                # embedding
                seqlen = len(x[m])
                h0 = self.embed_x(x[m],m)
                # forward path
                aa=self.l1f_x[m](F.dropout(h0, training=self.train))
                fh1 = torch.split(self.l1f_x[m](F.dropout(h0, training=self.train)), self.bsize, dim=0)
                fstate = self.make_initial_state(self.enc_hsize[m])
                h1f = []
                for h in fh1:
                    fstate = self.forward_one_step(h, fstate, m)
                    h1f.append(fstate['h1'])
                # backward path
                bh1 = torch.split(self.l1b_x[m](F.dropout(h0, training=self.train)), self.bsize, dim=0)
                bstate = self.make_initial_state(self.enc_hsize[m])
                h1b = []
                for h in reversed(bh1):
                    bstate = self.backward_one_step(h, bstate, m)
                    h1b.insert(0, bstate['h1'])
                # concatenation
                h1[m] = torch.cat([torch.cat((f,b), 1)
                              for f,b in six.moves.zip(h1f,h1b)], 0)
            else:
                # embedding only
                h1[m] = torch.tanh(self.embed_x(x[m],m))
        #
        return h1

    # Attention
    def attention(self, h, vh, s, s_qa, s_q):
        c = [None] * self.n_inputs

        for m in six.moves.range(self.n_inputs):
            bsize = self.bsize
            seqlen = h[m].data.shape[0] / bsize
            csize = h[m].data.shape[1]
            asize = self.att_size

            h_m = h[m].view(seqlen, bsize, csize)
            c[m] = h_m.mean(0)
        return c

    # Simple modality fusion
    def simple_modality_fusion(self, c, s, s_qa, s_q):
        g1 = self.l2d(s)
	g2 = self.l4d(s_qa)
	g3 = self.l5d(s_q)
        # for m in six.moves.range(self.n_inputs):
        #     g += self.lgd[m](F.dropout(c[m], training=self.train))
        for m in six.moves.range(self.n_inputs):
            if m==0:
                g = self.lgd[m](c[m])
            else:
                g += self.lgd[m](c[m])
        g = torch.cat((g, g1, g2, g3), 1)
        return F.tanh(g)

    # Decoder functions
    def embed_y(self, y_data):
        y = torch.tensor(y_data, dtype=torch.long)
        return self.emb_y(y.cuda())

    def decode_one_step(self, s, y, linear_y, linear_s):
        y_new = linear_y(y) + linear_s(s['h1'].cuda())
        # y_new = self.l1d_y(F.dropout(y, training=self.train)) + self.l1d_h(s['h1'].cuda())
        y_list = torch.split(y_new, self.dec_hsize,dim=1)
        y_list = list(y_list)
        c1 = F.tanh(y_list[0])*F.sigmoid(y_list[1])+s['c1'].cuda()*F.sigmoid(y_list[2])
        h1 = F.tanh(c1)*F.sigmoid(y_list[3])
        return {'c1': c1, 'h1': h1}

    def QA_decode(self, s, qa, linear_y, linear_s):
        y = self.embed_y(np.array([self.sos] * self.bsize, dtype=np.int32))
        s = self.decode_one_step(s, y, linear_y, linear_s)
        for i in six.moves.range(len(qa)):
            y = self.embed_y(qa[i])
            s = self.decode_one_step(s, y, linear_y, linear_s)
        return s

    def cross_entropy(self, s, s_qa, s_q, c, t_data):
        t = torch.tensor(t_data, dtype=torch.long)
        g = self.simple_modality_fusion(c,s,s_qa,s_q)
        y = self.l3d(F.dropout(g, training=self.train))
        return F.cross_entropy(y.cuda(), t.cuda(),ignore_index=-1)

    def classify(self, s, s_qa, s_q, c):
        g = self.simple_modality_fusion(c,s,s_qa,s_q)
        y = self.l3d(g)
        logp = F.log_softmax(y,dim=1)
        logp = logp.cpu().data.numpy()
        return np.argmax(logp, axis=1), logp


    # forward propagation routine
    def propagate(self, x, Q, A, predicted_context=False, train=True):
        self.bsize = x[0][0].shape[0]
        self.train = train
        loss = torch.zeros(1, dtype=torch.float).cuda()
        # encoder
        h1 = self.encode(x)
        vh1 = [self.atV[m](h1[m]) for m in six.moves.range(self.n_inputs)]
        # decoder
        QA_state = self.make_initial_state(self.dec_hsize)
        for i in six.moves.range(Q.shape[1]):
            question = Q[:, i, :]
            answer = A[:, i, :]
            Q_state = self.make_initial_state(self.dec_hsize)
            Q_state = self.QA_decode(Q_state, question, self.l1d_y_qq, self.l1d_h_qq)
            QA_state = self.QA_decode(QA_state, question, self.l1d_y_q, self.l1d_h_q)
	    
            dstate = self.make_initial_state(self.dec_hsize)
            y = self.embed_y(np.array([self.sos] * self.bsize, dtype=np.int32))
            dstate = self.decode_one_step(dstate, y, self.l1d_y, self.l1d_h)
            for j in six.moves.range(len(answer)):
                g = self.attention(h1, vh1, dstate['h1'], QA_state['h1'],Q_state['h1'])
                answerb = np.copy(answer[j])
                answerb[answerb == self.ignore_label] = -1
                loss_k = self.cross_entropy(dstate['h1'], QA_state['h1'], Q_state['h1'], g, answerb)
                loss = loss + loss_k
                if predicted_context == True:
                    pred, logp = self.classify(dstate['h1'], g)
                    y = self.embed_y(pred.astype(np.int32))
                else:
                    y = self.embed_y(answer[j])
                dstate = self.decode_one_step(dstate, y, self.l1d_y, self.l1d_h)
            QA_state = self.QA_decode(QA_state, answer, self.l1d_y_a, self.l1d_h_a)

        return loss

    # evaluation routine
    def evaluate(self, x, Q, A, predicted_context=False):
        self.bsize = x[0][0].shape[0]
        self.train = False
        loss = 0.
        hit = 0
        nlabels = 0
        # encoder
        h1 = self.encode(x) 
        vh1 = [self.atV[m](h1[m]) for m in six.moves.range(self.n_inputs)]
        # decoder
        QA_state = self.make_initial_state(self.dec_hsize)
        for i in six.moves.range(Q.shape[1]):
            question = Q[:, i, :]
            answer = A[:, i, :]
            Q_state = self.make_initial_state(self.dec_hsize)
            Q_state = self.QA_decode(Q_state, question, self.l1d_y_qq, self.l1d_h_qq)
            QA_state = self.QA_decode(QA_state, question, self.l1d_y_q, self.l1d_h_q)
	    
            dstate = self.make_initial_state(self.dec_hsize)
            y = self.embed_y(np.array([self.sos] * self.bsize, dtype=np.int32))
            dstate = self.decode_one_step(dstate, y, self.l1d_y, self.l1d_h)
            for j in six.moves.range(len(answer)):
                g = self.attention(h1, vh1, dstate['h1'], QA_state['h1'], Q_state['h1'])
                pred, logp = self.classify(dstate['h1'], QA_state['h1'], Q_state['h1'], g)
                flag = (answer[j] != self.ignore_label)
                loss -= np.sum(logp[six.moves.range(self.bsize), answer[j]] * flag)
                hit += np.sum((pred == answer[j]) * flag)
                nlabels += np.sum(flag)
                if predicted_context == True:
                    pred, logp = self.classify(dstate['h1'], QA_state['h1'], g)
                    y = self.embed_y(pred.astype(np.int32))
                else:
                    y = self.embed_y(answer[j])
                dstate = self.decode_one_step(dstate, y, self.l1d_y, self.l1d_h)
	    QA_state = self.QA_decode(QA_state, answer, self.l1d_y_a, self.l1d_h_a)
            

        return loss, hit, nlabels


    # generation routine
    def generate(self, x, Q, A, maxlen=100, beam=5, penalty=2.0, unk=0):
        final_logp=[]
        final_pred_out=[]
        self.bsize = 1
        self.train = False
        # encoder
        h1 = self.encode(x)
        vh1 = [self.atV[m](h1[m]) for m in six.moves.range(self.n_inputs)]
        # generator
        QA_state = self.make_initial_state(self.dec_hsize)
        for i in six.moves.range(Q.shape[1]):
            question = Q[:, i, :]
	    answer = A[:, i, :]
            Q_state = self.make_initial_state(self.dec_hsize)
            Q_state = self.QA_decode(Q_state, question, self.l1d_y_qq, self.l1d_h_qq)
            QA_state = self.QA_decode(QA_state, question, self.l1d_y_q, self.l1d_h_q)
	    
            dstate = self.make_initial_state(self.dec_hsize)
            y = self.embed_y(np.array([self.sos] * self.bsize, dtype=np.int32))
            dstate = self.decode_one_step(dstate, y, self.l1d_y, self.l1d_h)
            g = [None] * self.n_inputs
            hyplist = [([], 0., dstate)]
            comp_hyplist = []
            for l in six.moves.range(maxlen):
                new_hyplist = []
                min_id = 0
                for out, lp, st in hyplist:
                    g = self.attention(h1, vh1, st['h1'], QA_state['h1'], Q_state['h1'])
                    olab, logp = self.classify(st['h1'], QA_state['h1'], Q_state['h1'], g)
                    lp_vec = logp[0] + lp
                    # for o in six.moves.range(len(prob)):
                    if l > 0:
                        new_lp = lp_vec[self.eos] + penalty * (len(out) + 1)
                        comp_hyplist.append((out + [self.eos], new_lp))

                    for n, o in enumerate(np.argsort(lp_vec)[::-1]):
                        if n==0:
                            o1=0
                        if o == unk or o == self.eos:
                            continue
                        new_lp = lp_vec[o]
                        if len(new_hyplist) == beam:
                            if new_hyplist[min_id][1] < new_lp:
                                y = self.embed_y(np.array([o], dtype=np.int32))
                                new_st = self.decode_one_step(st, y, self.l1d_y, self.l1d_h)
                                new_hyplist[min_id] = (out + [o], new_lp, new_st)
                                min_id = min(enumerate(new_hyplist), key=lambda x: x[1][1])[0]
                            else:
                                break
                        else:
                            y = self.embed_y(np.array([o], dtype=np.int32))
                            new_st = self.decode_one_step(st, y, self.l1d_y, self.l1d_h)
                            new_hyplist.append((out + [o], new_lp, new_st))
                            if len(new_hyplist) == beam:
                                min_id = min(enumerate(new_hyplist), key=lambda x: x[1][1])[0]

                hyplist = new_hyplist

            if len(comp_hyplist) > 0:
                maxhyps = sorted(comp_hyplist, key=lambda x: -x[1])
                pred_out = [hyp[0] for hyp in maxhyps]
                logp = [hyp[1] for hyp in maxhyps]
            else:
                pred_out = [[]]
                logp = []
            QA_state = self.QA_decode(QA_state, answer, self.l1d_y_a, self.l1d_h_a)
            final_logp.append(logp)
            final_pred_out.append(pred_out)
        return final_pred_out, final_logp


    def forward(self, x, Q, A, predicted_context=False, istraining=True):
        if istraining:
            return self.propagate(x, Q, A, predicted_context=predicted_context, train=istraining)
        else:
            return self.evaluate(x, Q, A, predicted_context=predicted_context)

