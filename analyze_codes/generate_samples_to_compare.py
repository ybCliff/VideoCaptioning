import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import json
import os
import argparse
import torch
import shutil
import numpy as np
from pandas.io.json import json_normalize
import cv2
import sys

def idx_to_frameid(path, n_frames, index):
    length = len(os.listdir(path))
    bound = [int(i) for i in np.linspace(0, length, n_frames + 1)]
    index = (bound[index] + bound[index+1]) // 2
    return index + 1

def run(args, vid, sents):
    if not os.path.exists(args.examples_pth):
        os.makedirs(args.examples_pth)
    
    row = 1
    col = args.n_frames // row
    f, axarr = plt.subplots(row, col)
    for i in range(args.n_frames):
        ax = axarr[i]

        frames_pth = os.path.join(args.frames_pth, vid)
        frameid = idx_to_frameid(frames_pth, args.n_frames, i)

        pth = os.path.join(frames_pth, 'image_%05d.jpg'%frameid)

        img = cv2.imread(pth)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_yticks([])
        ax.set_xticks([])

    plt.subplots_adjust(left=0.02, bottom=None, right=0.98, top=0.99,
                wspace=None, hspace=None)

    fig = plt.gcf()
    fig.set_size_inches(7, 4)

    f.text(0.12, 0.24, 'A: ' + sents[0], fontsize=16)
    f.text(0.12, 0.12, 'B: ' + sents[1], fontsize=16)

    #plt.show()
    plt.savefig(os.path.join(args.examples_pth, '%s.jpg'%vid))

def run_all(args, vid, sents):
    if not os.path.exists(args.examples_pth):
        os.makedirs(args.examples_pth)
    
    row = 1
    col = args.n_frames // row
    f, axarr = plt.subplots(row, col)
    for i in range(args.n_frames):
        ax = axarr[i]

        frames_pth = os.path.join(args.frames_pth, vid)
        frameid = idx_to_frameid(frames_pth, args.n_frames, i)

        pth = os.path.join(frames_pth, 'image_%05d.jpg'%frameid)

        img = cv2.imread(pth)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_yticks([])
        ax.set_xticks([])

    plt.subplots_adjust(left=0.02, bottom=None, right=0.98, top=0.99,
                wspace=None, hspace=None)

    fig = plt.gcf()
    fig.set_size_inches(7, 4)
    fontsize = 12
    f.text(0.12, 0.36, 'GT  : ' + sents[0], fontsize=fontsize)
    f.text(0.12, 0.28, 'NA2E: ' + sents[1], fontsize=fontsize)
    f.text(0.12, 0.20, 'NA-B: ' + sents[2], fontsize=fontsize)
    f.text(0.12, 0.12, 'AR-B: ' + sents[3], fontsize=fontsize)
    f.text(0.12, 0.04, 'AR-B2: ' + sents[4], fontsize=fontsize)
    f.text(0.12, 0.8, 'Attr: ' + sents[5], fontsize=fontsize)


    #plt.show()
    plt.savefig(os.path.join(args.examples_pth, '%s.jpg'%vid))

import pickle
def main(args):
    captions = []
    assert len(args.pickle_to_load_captions) == 2
    for i in range(2):
        captions.append(pickle.load(open(args.pickle_to_load_captions[i], 'rb'))[0])

    keylist = list(captions[0].keys())
    if len(args.target):
        args.examples_pth += '_target%d'%len(args.target)
        candidate = sorted(args.target)
    elif args.all:
        args.examples_pth += '_all'
        candidate = sorted(keylist)
    else:
        candidate = np.random.choice(keylist, 1)

    gts = pickle.load(open("/home/yangbang/VideoCaptioning/MSRVTT/msrvtt_refs.pkl", 'rb'))
    ar = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/AR_topk_collect_results/msrvtt_1.pkl", 'rb'))
    ar2 = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/AR_topk_collect_results/msrvtt_1_ag.pkl", 'rb'))
    for key in candidate:
        if os.path.exists(os.path.join(args.examples_pth, '%s.jpg'%key)):
            continue
        sents = []
        if args.all: sents.append(gts[key][0]['caption'])
        sents.append(captions[0][key][-1])
        sents.append(captions[1][key][-1])
        if args.all: 
            sents.append(ar[key][0]['caption'])
            sents.append(ar2[key][0]['caption'])
            sents.append(captions[0][key][0].replace('<mask>', '<m>'))
            run_all(args, key, sents)
        else:
            run(args, key, sents)


if __name__ == '__main__':
    #torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_to_load_captions', type=str, nargs='+', default=[
            "/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/MSRVTT_nv_AEmp_i5b6a135.pkl",
            "/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/MSRVTT_mp_mp_i5b6a135.pkl"
        ])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--frames_pth', type=str, default='/home/yangbang/VideoCaptioning/MSRVTT/all_frames/')
    parser.add_argument('--n_frames', type=int, default=4)
    parser.add_argument('--examples_pth', type=str, default='./qualitative_examples')
    parser.add_argument('--all', default=False, action='store_true')
    parser.add_argument('--target', nargs='+', type=str, default=[])

    args = parser.parse_args()
    main(args)


