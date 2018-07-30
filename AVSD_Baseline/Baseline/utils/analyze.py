#!/usr/bin/python
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import sys
import json
import re

# get arguments and options
parser = argparse.ArgumentParser()
parser.add_argument('--vidfile', '-f', default='', type=str,
                    help='Video directory')
parser.add_argument('--metric', '-m', default='Bleu_4', type=str,
                    help='Metric (Bleu_?, METEOR, ROUGE_L, or CIDEr)')
parser.add_argument('--vidmap', '-v', default='data/youtube/dict_youtube_mapping.json',
                    type=str, help='video ID map file')
parser.add_argument('--output', '-o', default='', type=str, help='output file (.csv)')
parser.add_argument('reference', metavar='REFFILE', type=str, nargs='?',
                   help='data set file in json format as {dir:id,...}')
parser.add_argument('result1', metavar='RESULT1', type=str, nargs='?',
                   help='result1 to be compared (.json)')
parser.add_argument('result2', metavar='RESULT2', type=str, nargs='?',
                   help='result2 to be compared (.json)')

args = parser.parse_args()

# read video-id to url-id mapping
if args.vidmap != '':
    vidmap = json.load(open(args.vidmap,'r'))
    vidmap_inv = {}
    for k,v in vidmap.iteritems():
        vidmap_inv[v] = k
else:
    vidmap_inv = None

# make id to video-id
refdata = json.load(open(args.reference,'r'))
idmap = {}
for data in refdata['images']:
    idmap[data['id']] = data['name']

# id to caption for result1
res1cap = {}
for data in json.load(open(args.result1,'r')):
    res1cap[data['image_id']] = data['caption']
# id to caption for result2
res2cap = {}
for data in json.load(open(args.result2,'r')):
    res2cap[data['image_id']] = data['caption']
 
# calculate evaluation metrics
coco = COCO(args.reference)
cocoRes1 = coco.loadRes(args.result1)
cocoEval1 = COCOEvalCap(coco, cocoRes1)
cocoEval1.evaluate()
cocoRes2 = coco.loadRes(args.result2)
cocoEval2 = COCOEvalCap(coco, cocoRes2)
cocoEval2.evaluate()

# store all data
report = []
for (key,value1),(key_,value2) in zip(cocoEval1.imgToEval.iteritems(),cocoEval2.imgToEval.iteritems()):
    assert key == key_, "Keys do not match"
    urlid = vidmap_inv[idmap[key]]
    videofile = re.sub('{VID}', urlid, args.vidfile)
    score1 = value1[args.metric]
    score2 = value2[args.metric]
    report.append((key,idmap[key],videofile,res1cap[key],res2cap[key],score1,score2))

# report summary
if args.output != '':
    fo = open(args.output,'w')
else:
    fo = sys.stdout

report = sorted(report, key=lambda s:s[6]-s[5])
for key,vid,vidfile,cap1,cap2,score1,score2 in report:
    print >> fo, '%s,%s,%s,%.3f,%.3f,%s' % (vid,cap1,cap2,score1,score2,vidfile)
    #print >> fo, '%s\t%s\t%s\t%.3f\t%.3f\t%s' % (vid,cap1,cap2,score1,score2,vidfile)

fo.close()
