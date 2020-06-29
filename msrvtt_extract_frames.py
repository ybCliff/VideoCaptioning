import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import pretrainedmodels
from pretrainedmodels import utils

C, H, W = 3, 224, 224


def extract_frames(video, dst, fps='3', vframes='50'):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=iw:-1", # input file
                                   '-r', fps, #fps 5
                                   '-vframes', vframes,
                                   '{0}/%05d.png'.format(dst)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)


def run(params):
    video = params['video_name']
    video_id = video.split("/")[-1].split(".")[0]

    dst = video_id
    extract_frames(params['video_path'], dst)
    '''
    image_list = sorted(glob.glob(os.path.join(dst, '*.png')))
    #print(image_list)
    samples = np.round(np.linspace(
            0, len(image_list) - 1, params['n_frame_steps']))
    s = [int(n) for n in samples]
    for i in range(len(image_list)):
        if i not in s:
            os.remove(image_list[i])
    '''


import pickle
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", dest='video_name', type=str, default='video0')
    parser.add_argument("--video_path", dest='video_path', type=str,
                        default="/home/yangbang/VideoCaptioning/MSRVTT/all_videos/", help='path to video dataset')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=30,
                        help='how many frames to sampler per video')
    
    refs = pickle.load(open('/home/yangbang/VideoCaptioning/MSRVTT/msrvtt_refs.pkl', 'rb'))

    args = parser.parse_args()
    for item in refs[args.video_name]:
        print(item['caption'])
    args.video_name = args.video_name + '.mp4'
    params = vars(args)
    params['video_path'] = os.path.join(params['video_path'], params['video_name'])
    run(params)
