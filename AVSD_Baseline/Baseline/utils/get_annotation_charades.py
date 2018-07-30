#!/usr/bin/python
# coding: utf-8

import json
import sys
import re

if len(sys.argv) < 4:
    print 'usage:', sys.argv[0], 'CAP.json testset.json output.json'
    sys.exit(1)
capfile=sys.argv[1]
testsetfile=sys.argv[2]
output=sys.argv[3]

data = {}
data['info'] = {}
data['licenses'] = []
data['type'] = 'captions'

annos = []
images = []
image_id=1
cap_id=1
captions=json.load(open(capfile,'r'))
testset=json.load(open(testsetfile,'r'))

vidset = set()
for idx in testset:
    vid,cid = idx.split('_')
    if vid not in vidset:
        vidset.add(vid)
        for cap in captions[vid]:
            sent = cap['tokenized']
            sent = re.sub('’',"'", sent)
            sent = re.sub('"|\t','', sent)
            annos.append({"image_id": image_id, "id": cap_id, "caption": sent})
            cap_id += 1
        images.append({"name": vid, "id": image_id})
        image_id += 1
        
data['annotations'] = annos
data['images'] = images

json.dump(data, open(output,'w'), indent=4)

