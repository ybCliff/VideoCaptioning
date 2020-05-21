import sys
sys.path.append("..")
sys.path.append(".")
import numpy as np
import math
import pickle
from misc.cocoeval import suppress_stdout_stderr, COCOScorer, COCOBLEUScorer
'''
sent, score = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/all/MSRVTT_nv_AEmp_i5b6a135.pkl", 'rb'))
scorer = COCOScorer()
gts = pickle.load(open("/home/yangbang/VideoCaptioning/MSRVTT/msrvtt_refs.pkl", 'rb'))

B = 6
keylist = sent.keys()
res = {}
res2 = {}
for k in keylist:
    res[k] = []
    res2[k] = []
with suppress_stdout_stderr():
    for i in range(B):
        samples = {}
        for key in keylist:
            samples[key] = []
            samples[key].append({'image_id': key, 'caption': sent[key][5::6][i]})
        valid_score, detail_scores = scorer.score(gts, samples, samples.keys())
        for k in keylist:
            res[k].append(detail_scores[k]['CIDEr'] + detail_scores[k]['METEOR'])
            res2[k].append(detail_scores[k]['METEOR'])

with suppress_stdout_stderr():
    samples = {}
    for key in keylist:
        samples[key] = []
        index = np.array(res[key]).argmax()
        samples[key].append({'image_id': key, 'caption': sent[key][5::6][index]})
    valid_score, detail_scores = scorer.score(gts, samples, samples.keys())
print(valid_score)

with suppress_stdout_stderr():
    samples = {}
    for key in keylist:
        samples[key] = []
        index = np.array(res2[key]).argmax()
        samples[key].append({'image_id': key, 'caption': sent[key][5::6][index]})
    valid_score, detail_scores = scorer.score(gts, samples, samples.keys())
print(valid_score)
'''
sent = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/AR_topk_collect_results/msrvtt_5.pkl", 'rb'))
scorer = COCOScorer()
gts = pickle.load(open("/home/yangbang/VideoCaptioning/MSRVTT/msrvtt_refs.pkl", 'rb'))

B = 5
keylist = sent.keys()
res = {}
res2 = {}
for k in keylist:
    res[k] = []
    res2[k] = []
with suppress_stdout_stderr():
    for i in range(B):
        samples = {}
        for key in keylist:
            samples[key] = []
            samples[key].append({'image_id': key, 'caption': sent[key][i]['caption']})
        valid_score, detail_scores = scorer.score(gts, samples, samples.keys())
        for k in keylist:
            res[k].append(detail_scores[k]['CIDEr'] + detail_scores[k]['METEOR'])
            res2[k].append(detail_scores[k]['METEOR'])

with suppress_stdout_stderr():
    samples = {}
    for key in keylist:
        samples[key] = []
        index = np.array(res[key]).argmax()
        samples[key].append({'image_id': key, 'caption': sent[key][index]['caption']})
    valid_score, detail_scores = scorer.score(gts, samples, samples.keys())
print(valid_score)

with suppress_stdout_stderr():
    samples = {}
    for key in keylist:
        samples[key] = []
        index = np.array(res2[key]).argmax()
        samples[key].append({'image_id': key, 'caption': sent[key][index]['caption']})
    valid_score, detail_scores = scorer.score(gts, samples, samples.keys())
print(valid_score)
