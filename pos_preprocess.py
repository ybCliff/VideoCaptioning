from collections import Counter
import json
import nltk
import h5py
import numpy as np
'''
caps = json.load(open("/home/yangbang/VideoCaptioning/MSRVTT/caption_pad_mask_2.json"))

words = {}
for i in range(10000):
    vid = 'video%d'%i
    for cap in caps[vid]['final_captions']:
        #print()
        sent = cap[1:-1]
        pos_tags = nltk.pos_tag(sent)
        for tag in pos_tags:
            if tag[1] not in words.keys():
                words[tag[1]] = []
            words[tag[1]].append(tag[0])

rec = []
for k,v in words.items():
    if 'NN' in k:
        rec += v
res = Counter(rec)
print(len(res))
dim = 1001
wl = list(dict(res.most_common(dim)).keys())

db = h5py.File('/home/yangbang/VideoCaptioning/MSRVTT/feats/nn_%d.hdf5'%dim, 'a')
w2id = {}
id2w = {}

i = 0
for w in wl:
    if w == '<unk>':
        continue
    w2id[w] = i
    id2w[i] = w
    i += 1

for i in range(10000):
    vid = 'video%d'%i
    data = np.zeros(dim-1)
    for cap in caps[vid]['final_captions']:
        #print()
        sent = cap[1:-1]
        for w in sent:
            if w in w2id.keys():
                data[w2id[w]] = 1
    db[vid] = data

db.close()
with open("/home/yangbang/VideoCaptioning/MSRVTT/nn_%d.json"%dim, 'w') as f:
    json.dump({'w2id':w2id, 'id2w':id2w}, f)

'''

caps = json.load(open("/home/yangbang/VideoCaptioning/Youtube2Text/caption_pad_mask_0.json"))

words = {}
for i in range(1970):
    vid = 'video%d'%i
    for cap in caps[vid]['final_captions']:
        #print()
        sent = cap[1:-1]
        pos_tags = nltk.pos_tag(sent)
        for tag in pos_tags:
            if tag[1] not in words.keys():
                words[tag[1]] = []
            words[tag[1]].append(tag[0])

rec = []
for k,v in words.items():
    if 'NN' in k:
        rec += v
res = Counter(rec)
print(len(res))
dim = 1001
wl = list(dict(res.most_common(dim)).keys())

db = h5py.File('/home/yangbang/VideoCaptioning/Youtube2Text/feats/nn_%d.hdf5'%dim, 'a')
w2id = {}
id2w = {}

i = 0
for w in wl:
    if w == '<unk>':
        continue
    w2id[w] = i
    id2w[i] = w
    i += 1

for i in range(1970):
    vid = 'video%d'%i
    data = np.zeros(dim-1)
    for cap in caps[vid]['final_captions']:
        #print()
        sent = cap[1:-1]
        for w in sent:
            if w in w2id.keys():
                data[w2id[w]] = 1
    db[vid] = data

db.close()
with open("/home/yangbang/VideoCaptioning/Youtube2Text/nn_%d.json"%dim, 'w') as f:
    json.dump({'w2id':w2id, 'id2w':id2w}, f)

