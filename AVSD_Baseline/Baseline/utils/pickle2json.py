#!/usr/bin/python
"""A pickle-to-json converter
   Copyright 2016 Mitsubishi Electric Research Labs
"""
import argparse
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('--indir', '-i', default='', type=str,
                     help='Specify input directory')
parser.add_argument('--outdir', '-o', default='', type=str,
                     help='Specify output directory')
parser.add_argument('--indent', default=4, type=int,
                     help='Indent size of output json files')
parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                    help='Input pickle files')
args = parser.parse_args()

for fn in args.input:
    out = fn.replace(".pkl", ".json")
    if args.indir != '':
        fn = args.indir + '/' + fn
    if args.outdir != '':
        out = args.outdir + '/' + out
    print '%s -> %s' % (fn, out)
    data = pickle.load(open(fn,'r'))
    json.dump(data, open(out,'w'), indent=args.indent)
print 'done'
