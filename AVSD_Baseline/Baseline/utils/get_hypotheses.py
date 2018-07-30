#!/usr/bin/python

import json
import sys
import re

if len(sys.argv) < 3:
    print 'usage:', sys.argv[0], 'hypfile output.json'
    sys.exit(1)
hypfile=sys.argv[1]
output=sys.argv[2]

annos = []
image_id=1
for line in open(hypfile,'r').readlines():
    if re.match(r'^HYP(\[1\])?:', line):
        line = line.strip()
        line = re.sub(r'HYP(\[1\])?:|<eos>|\([\d\.\- ]+\)|,|\.', '', line)
        line = re.sub(r'\s+',' ', line).strip()
        annos.append({'image_id': image_id, 'caption': line})
        image_id += 1
        
json.dump(annos, open(output,'w'), indent=4)

