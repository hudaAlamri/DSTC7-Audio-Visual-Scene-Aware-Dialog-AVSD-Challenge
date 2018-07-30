"""I3D feature extration using a tensorflow model.
   Copyright 2018 Mitsubishi Electric Research Labs
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import tensorflow as tf
import time
import os
import scipy.io as sio
import skimage.io
from skimage.transform import rescale, resize, downscale_local_mean
from random import randint
import cv2
import i3d
from i3d import Unit3D
import sonnet as snt
import skvideo.io
import pickle
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--input', default='data/Charades_v1_rgb', type=str,
                    help='Directory that includes image files')
parser.add_argument('--net_output', default='Mixed_5c',
                    type=str, help="layer used as output features")
parser.add_argument('--feature_dim', '-f', default=2048, type=int,
                    help='output feature dimension')
parser.add_argument('--model_path', default='data/i3d_model/data/checkpoints/rgb_imagenet', type=str, help='model path')
parser.add_argument('--stride', default=4, type=int, help='stride of frame features')
parser.add_argument('--output',default='data/Charades/i3d_rgb', type=str,
                       help='output pickle file of feature vectors')
parser.add_argument('--seq_length', default=16, type=int, help='window size of frame features')
args = parser.parse_args()

_IMAGE_SIZE = 224
_NUM_CLASSES = 400


def train():
    print (args.model_path)
    model_path = args.model_path
    pose_net_path = os.path.join(model_path, 'model.ckpt')
    tf.reset_default_graph()
    with tf.variable_scope('RGB'):
        rgb_input = tf.placeholder(tf.float32, [None, args.seq_length, _IMAGE_SIZE, _IMAGE_SIZE, 3])
        rgb_y = tf.placeholder(tf.float32, [None, _NUM_CLASSES])
        lr = tf.placeholder("float")
        drop_out_prob = tf.placeholder("float")

        i3d_model = i3d.InceptionI3d(num_classes=_NUM_CLASSES, final_endpoint='Mixed_5c')
        net, end_points = i3d_model(rgb_input, is_training=False, dropout_keep_prob=drop_out_prob)

    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable

    tf_config = tf.ConfigProto()
    restorer = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    with tf.Session(config=tf_config) as sess:
        restorer.restore(sess, pose_net_path)
        lr_s = 0.0001
        drop_out = 1
        save_folder = args.output
        root_folder = args.input
        num_seq = len(os.listdir(root_folder))
        for f1 in os.listdir(root_folder):           
            seq = os.listdir(os.path.join(root_folder, f1))
            f_exit = os.listdir(save_folder)
            if f1 not in f_exit:
                os.mkdir(os.path.join(save_folder, f1))
            else:
                if os.listdir(os.path.join(save_folder, f1)) !=[]:
                    continue	    
            num_frame = len(seq)
            if num_frame < args.seq_length:
                print("There should be at least",args.seq_length," frames")
            num_sample = num_frame//args.stride
            features = np.zeros(shape=[num_sample, args.feature_dim])           
            for i in range(0, num_sample):                
                Start_f = i*args.stride + 1
                input = np.zeros(shape=[1, args.seq_length, _IMAGE_SIZE, _IMAGE_SIZE, 3])
                gth_label = np.zeros(shape=[1, _NUM_CLASSES])
                for j in range(0, args.seq_length):
                    pick_f = Start_f + j
                    if pick_f > num_frame:
                        pick_f = Start_f
                    im = cv2.imread(os.path.join(root_folder, f1, (f1 + '-' + ("%06d" % pick_f) + '.jpg')))
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)                    
                    im = cv2.resize(im, (_IMAGE_SIZE, _IMAGE_SIZE))                    
                    im = (im - 128)/128
                    input[:, j, :, :, :] = im
                gth_label[0] = 1
                feed_dict = {
                    rgb_input: input,
                    rgb_y: gth_label,
                    lr: lr_s,
                    drop_out_prob: drop_out
                }
                logits, net_feature = sess.run([net, end_points], feed_dict)
                Mix5c = net_feature[args.net_output]
                feature = Mix5c.mean(axis=(2,3))
                feature = feature.reshape((1, 2048))
                features[i, :] = feature
            pickle.dump(features, open(os.path.join(save_folder, f1) + '/feature.pkl', 'wb'), 2)


def main(argv=None):
  train()


if __name__ == '__main__':
    tf.app.run()
