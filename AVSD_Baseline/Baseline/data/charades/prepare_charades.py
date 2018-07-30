#!/usr/bin/env python
"""Image feature extration using a caffe model
   Copyright 2016 Mitsubishi Electric Research Labs
"""

import argparse
import logging
import sys
import time
import os
import re
import glob
import json
import six

import numpy as np

import pickle
from scipy.misc import imread, imresize, imsave
import skimage.transform
import scipy.io as sio
from sklearn import preprocessing


# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging.warn("The value of --feature-dim is ignored. The feature will be saved in its original shape.")
    # output immediately
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_map', default='', type=str)
    parser.add_argument('--feature_dir', default='i3d_rgb',
                        type=str, help="feature folder")
    parser.add_argument('--output', default='',
                        type=str, help="output name")
    parser.add_argument('--skip', default='4',
                        type=int, help="downsample")
    args = parser.parse_args()
    args.id_map='dict_charades_mapping.json'
    args.output='charades2text_i3d_rgb_features_stride16.pkl'

    args.offset = 3
    args.id_pattern = '/([^/\s]+)/jpg'
    if args.id_map != '':
        print("loading image-id mapping", args.id_map)
        idmap = json.load(open(args.id_map, 'r'))
        print("done")

    idpat = re.compile(args.id_pattern)
    output = {}
    for path in os.listdir(args.feature_dir):
        vid = path[0:5]
        if vid in idmap:
            vid = idmap[vid]
        else:
            raise RuntimeError('Unknown Video ID ' + vid)
        y_feature = np.load(open(args.feature_dir + '/' + path, 'r'))
        feature_downsample = y_feature[::args.skip]
        print('ID:', vid, '-- processing', y_feature.shape[0], 'images in', path)
        output[vid] = feature_downsample

    print("saving features to", args.output)
    pickle.dump(output, open(args.output, 'wb'), 2)
    print("done")
