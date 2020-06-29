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
            'info_corpus', 'vocab_size',
            'with_category', 'num_category', 
            'decoder_type', 'dataset', 'n_frames', 'max_len', 'top_down'
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

    loader_list = []
    for item in opt_list:
        loader = get_loader(item, mode=opt['em'], print_info=False, specific=opt['specific'])
        loader_list.append(loader)

    vocab = loader_list[0].dataset.get_vocab()

    metric = run_eval_ensemble(opt, opt_list, model_list, None, loader_list, vocab, device, 
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
            "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/GRU_LSTM/ADD0_WA0_EBN1_SS1_WC0_I_pan_lite/best/0036_225112_236269_235943_235829_233713.pth.tar",
            "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/GRU_LSTM/ADD0_WA0_EBN1_SS1_WC0_M-I_pan_full/best/0045_224531_234905_234775_234476_233589.pth.tar",
            "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/GRU_LSTM/ADD0_WA0_EBN1_SS1_WC0_A-M-I_pan_lite_full/best/0049_232514_237815_238561_241098_239263.pth.tar",
        ])
    parser.add_argument('--names', nargs='+', type=str, default=[
            "lite",
            "full",
            'lf',
        ])
    opt = parser.parse_args()

    main(opt)
