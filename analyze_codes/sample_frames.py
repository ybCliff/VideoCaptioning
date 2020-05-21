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

def idx_to_frameid(path, n_frames):
    length = len(os.listdir(path))
    bound = [int(i) for i in np.linspace(0, length, n_frames + 1)]
    index = []
    for i in range(n_frames):
        index.append((bound[i] + bound[i+1])// 2) 
    return [item+1 for item in index]

#frames_pth = '/home/yangbang/VideoCaptioning/Youtube2Text/all_frames/'
#low, high = 1300, 1970
frames_pth = '/home/yangbang/VideoCaptioning/MSRVTT/all_frames/'
low, high = 7010, 10000
n_frames = 8
#output_pth = './MSVD_frames'
output_pth = './MSRVTT_frames'
for i in range(low, high):
    vid = 'video%d'%i
    pth = os.path.join(frames_pth, vid)
    out = os.path.join(output_pth, vid)
    if not os.path.exists(out):
        os.makedirs(out)
    fid = idx_to_frameid(pth, n_frames)

    for _id in fid:
        img_name = 'image_%05d.jpg'%_id
        shutil.copy(os.path.join(pth, img_name), os.path.join(out, img_name))

