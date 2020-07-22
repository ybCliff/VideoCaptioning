''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import os, json
from models import get_model
from misc.run import get_loader, run_eval, get_forword_results
from sklearn.decomposition import PCA
from sklearn import manifold
import numpy as np
import os

import torch.nn.functional as F
import matplotlib.pyplot as plt
import json

def plot_several_category(data, category, specific=[3, 4, 9, 16, 17]):
    def cal_dis(x, y, category, num_category=20):
        ave_dis = []
        for s in range(num_category):
            idx = []
            for i in range(x.shape[0]):
                if s != -1 and category[i] != s:
                    continue
                idx.append(i)
            cluster = torch.stack([torch.from_numpy(x[idx]), torch.from_numpy(y[idx])], dim=1)
            center = cluster.mean(0).unsqueeze(0)
            ave_dis.append(F.pairwise_distance(center, cluster, p=2).view(-1))
        ave_dis = torch.cat(ave_dis, 0).mean(0)
        print(ave_dis)

    x = data[:, 0]
    y = data[:, 1]
    color = ['r', 'g', 'b', 'purple', 'orange', 'mediumturquoise', 'orange']
    label = ['sports', 'news', 'vehicles', 'food', 'cooking']
    for j, s in enumerate(specific):
        idx = []
        for i in range(x.shape[0]):
            if s != -1 and category[i] != s:
                continue
            idx.append(i)
        plt.scatter(x[idx], y[idx], c=color[j], label=label[j])

    cal_dis(x, y, category)
    #plt.legend(fontsize=18)
    plt.xticks([])
    plt.yticks([])
    #plt.xlabel('(a) image modality', fontsize=20)
    plt.xlabel('(b) audio modality', fontsize=20)
    plt.subplots_adjust(left=0.02, right=0.99, bottom=0.07, top=0.99)
    #plt.savefig('./I.png')
    plt.savefig('./A.png')
    plt.show()


def main(opt):
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')
    opt_pth = os.path.join(opt.model_path, 'opt_info.json')
    option = json.load(open(opt_pth, 'r'))
    option.update(vars(opt))
    # print(option)

    model = get_model(option)
    checkpoint = torch.load(os.path.join(opt.model_path, 'best', opt.model_name))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    for key in ['info_corpus', 'reference', 'feats_i', 'feats_m', 'feats_a']:
        if isinstance(option[key], list):
            for i in range(len(option[key])):
                option[key][i] = option[key][i].replace('/home/yangbang/VideoCaptioning', '/Users/yangbang/Desktop/VC_data')
        else:
            option[key] = option[key].replace('/home/yangbang/VideoCaptioning', '/Users/yangbang/Desktop/VC_data')

    loader = get_loader(option, mode=opt.em, print_info=False, specific=opt.specific)
    vocab = loader.dataset.get_vocab()

    if opt.oe:
        metric = run_eval(option, model, None, loader, vocab, device,
                          json_path=opt.json_path, json_name=opt.json_name, print_sent=opt.print_sent, no_score=opt.ns,
                          save_videodatainfo=opt.sv)
        print(metric)
    else:
        encoder_outputs = []
        category = []
        for data in tqdm(loader, ncols=150, leave=False):
            with torch.no_grad():
                results, cate, _, _ = get_forword_results(option, model, data, device=device, only_data=True)
                encoder_outputs.append(results['enc_output'][0])
                category.append(cate)
        encoder_outputs = torch.cat(encoder_outputs, dim=0).cpu().numpy()
        category = torch.cat(category, dim=0).view(-1).cpu().numpy()
        print(encoder_outputs.shape, category.shape)

        encoder_outputs = encoder_outputs.mean(1)
        pca = manifold.TSNE(n_components=opt.pca)
        data = pca.fit_transform(encoder_outputs)
        plot_several_category(data, category)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model_path', type=str,
                        default="./A")

    parser.add_argument('-model_name', default='0029_141016_144452_145352_146429_143949.pth.tar', type=str)
    #0035_172635_175599_178757_180008_178418.pth.tar
    #0029_141016_144452_145352_146429_143949.pth.tar
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-beam_alpha', type=float, default=1.0)
    parser.add_argument('-batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('-topk', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-max_len', type=int, default=30)
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
    parser.add_argument('-pca', default=2, type=int)
    parser.add_argument('--pca_path', default='./pca_results')
    parser.add_argument('-pn', '--pca_name', default='')
    parser.add_argument('-mean', default=False, action='store_true')
    parser.add_argument('-all', default=False, action='store_true')

    parser.add_argument('-plot', default=False, action='store_true')
    parser.add_argument('-t', default=0, type=int)
    parser.add_argument('-length_id', default=0, type=int)
    parser.add_argument('-gate_id', default=0, type=int)

    parser.add_argument('-oe', default=False, action='store_true')
    parser.add_argument('-dfi', '--dummy_feats_i', default=False, action='store_true')
    parser.add_argument('-dfm', '--dummy_feats_m', default=False, action='store_true')
    parser.add_argument('-dfa', '--dummy_feats_a', default=False, action='store_true')
    opt = parser.parse_args()

    main(opt)

