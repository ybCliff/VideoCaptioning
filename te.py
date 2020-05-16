''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import os,json
from misc.run import get_model, get_loader, run_eval_ensemble
from sklearn.decomposition import PCA
from sklearn import manifold
import numpy as np
import os

import torch.nn.functional as F
import matplotlib.pyplot as plt
import json

def main(opt):
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')
    opt = vars(opt)
    checklist = [
            'info_json', 'caption_json', 'next_info_json', 'all_caption_json', 'input_json',
            'feats_a', 'feats_m', 'feats_i',
            'dim_a', 'dim_m', 'dim_i',
            'with_category', 'num_category', 
            'decoder_type', 'dataset', 'n_frames', 'max_len'
            ]

    def check(d, other_d, key):
        value = other_d[key]
        if d.get(key, False):
            assert d[key] == value, "%s %s %s" %(key, str(d[key]), str(value))
        return value

    opt_list = []
    model_list = []
    checkpoint_paths = [item.split('/best/')[0] for item in opt['ensemble_checkpoint_paths']]
    pretrained_paths = opt['ensemble_checkpoint_paths']
    
    for i, pth in enumerate(checkpoint_paths):
        info = json.load(open(os.path.join(pth, 'opt_info.json')))
        opt_list.append(info)
        checkpoint = torch.load(pretrained_paths[i])
        model = get_model(info)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model_list.append(model)
        print(checkpoint['test_result'])

    for info in opt_list:
        for k in checklist:
            opt[k] = check(opt, info, k)

    names = opt['names']
    names.insert(0, opt['dataset'])     
    pth = os.path.join(opt['results_path'], opt['em'], '_'.join(names))
    if not os.path.exists(pth): os.makedirs(pth)

    with open(os.path.join(pth, 'ensemble_opt.json'), 'w') as f:
        json.dump(opt, f)

    loader = get_loader(opt, mode=opt['em'], print_info=False, specific=opt['specific'])
    vocab = loader.dataset.get_vocab()

    metric = run_eval_ensemble(opt, opt_list, model_list, None, loader, vocab, device, 
        json_path=opt['json_path'], json_name=opt['json_name'], 
        print_sent=opt['print_sent'], no_score=opt['ns'], analyze=True)
    print(metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-beam_alpha', type=float, default=1.0)
    parser.add_argument('-batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('-topk', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    #parser.add_argument('-max_len', type=int, default=30)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-em', type=str, default='test')
    parser.add_argument('-print_sent', action='store_true')
    parser.add_argument('-func', type=int, default=1)
    parser.add_argument('-json_path', type=str, default='')
    parser.add_argument('-json_name', type=str, default='')
    parser.add_argument('-ns', default=False, action='store_true')
    parser.add_argument('-sv', default=False, action='store_true')
    parser.add_argument('-rgi', '--return_gate_info', default=False, action='store_true')
    parser.add_argument('-specific', default=-1, type=int)

    parser.add_argument('--results_path', type=str, default='./ensemble_results')
    parser.add_argument('--ensemble_checkpoint_paths', nargs='+', type=str, default=[
            #"/home/yangbang/VideoCaptioning/1107save/MSRVTT/GRU_LSTM/AMI_Seed0_EBN_WC20_SS1_0_75_base_Ami/best/0046_179013_186663_189567_189993_189990.pth.tar",
            #"/home/yangbang/VideoCaptioning/1107save/MSRVTT/GRU_LSTM/AMI_Seed0_EBN_WC20_SS1_0_75_base_Mai/best/0044_179479_184734_188399_188466_187095.pth.tar",
            #"/home/yangbang/VideoCaptioning/1107save/MSRVTT/GRU_LSTM/AMI_Seed0_EBN_WC20_SS1_0_75_base_Iam/best/0042_177702_184338_187698_189578_189191.pth.tar",

            #"/home/yangbang/VideoCaptioning/1107save/MSRVTT/GRU_LSTM/A_Seed0_EBN_WC20_SS1_0_75_base/best/0034_138964_144352_145003_144404_143001.pth.tar",
            #"/home/yangbang/VideoCaptioning/1107save/MSRVTT/GRU_LSTM/I_Seed0_EBN_WC20_SS1_0_75_base/best/0030_171526_173378_175799_176511_176010.pth.tar",
            #"/home/yangbang/VideoCaptioning/1107save/MSRVTT/GRU_LSTM/M_Seed0_EBN_WC20_SS1_0_75_base/best/0035_171117_173436_177300_178115_176612.pth.tar",
            "/home/yangbang/VideoCaptioning/1107save/Youtube2Text/GRU_LSTM/MI_Seed0_EBN_SS1_0_75_base_Mi/best/0034_237815_249709_251475_249702_247184.pth.tar",
            #"/home/yangbang/VideoCaptioning/1107save/Youtube2Text/GRU_LSTM/MI_Seed0_EBN_SS1_0_75_base_Im/best/0036_242729_256354_256557_258267_258265.pth.tar"
            "/home/yangbang/VideoCaptioning/1107save/Youtube2Text/GRU_LSTM/MI_Seed0_EBN_SS1_0_75_base_Im/best/0015_238692_252909_253725_253461_251313.pth.tar"
        ])
    parser.add_argument('--names', nargs='+', type=str, default=[
            "Ami",
            "Mai",
            "Iam"
        ])
    opt = parser.parse_args()

    main(opt)
