''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import os,json
from models import get_model
from misc.run import get_loader, run_eval, get_forword_results
from sklearn.decomposition import PCA
from sklearn import manifold
import numpy as np
import os

import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
def plot(mode, jsonfile, path, filename, time, specific):
    info = json.load(open(jsonfile))['id_to_category']
    #info = json.load(open("/home/yangbang/VideoCaptioning/MSRVTT/hidden_topic_mining/tc10s10/20_15/annotation.json"))['id_to_category']
    if mode == 'validate':
        start = 6513
    else:
        start = 7010

    data = np.load(os.path.join(path, filename))
    if time == -1:
        print("======= mean =======")
        x = data[:, 0]
        y = data[:, 1]
    else:
        x = data[:, time, 0]
        y = data[:, time, 1]
    color = []
    idx = []
    for i in range(x.shape[0]):
        category = info[str(i + start)]
        #category = int(info['video' + str(i + start)])
        value = category * 0.05
        if specific != -1 and category != specific:
            continue
        idx.append(i)
        color.append(value)

    plt.scatter(x[idx], y[idx], c=color)
    plt.show()

def plot_several_category(mode, jsonfile, path, filename, time, 
    specific=[
        3, 
        4, 
        9, 
        #11, 
        #14, 
        16, 
        17
        ], n_frames=8):
    
    def cal_dis(x, y, num_category=20, all=False, n_frames=8):
        ave_dis = []
        for s in range(num_category):
            idx = []
            for i in range(x.shape[0]):
                category = info[str((i // n_frames if all else i) + start)]
                #category = int(info['video' + str(i + start)])
                value = category * 0.05
                if s != -1 and category != s:
                    continue
                idx.append(i)
            cluster = torch.stack([torch.from_numpy(x[idx]), torch.from_numpy(y[idx])], dim=1)
            center = cluster.mean(0).unsqueeze(0)
            ave_dis.append(F.pairwise_distance(center, cluster, p=2).view(-1))
        ave_dis = torch.cat(ave_dis, 0).mean(0)
        print(ave_dis)

    info = json.load(open(jsonfile))['id_to_category']
    #info = json.load(open("/home/yangbang/VideoCaptioning/MSRVTT/hidden_topic_mining/tc10s10/20_15/annotation.json"))['id_to_category']
    if mode == 'validate':
        start = 6513
    else:
        start = 7010

    data = np.load(os.path.join(path, filename))
    if time == -1:
        print("======= mean =======")
        #x = data[:, 0]
        #y = data[:, 1]
        x = [data[:2990, 0], data[2990:, 0]]
        y = [data[:2990, 1], data[2990:, 1]]
    elif time == -2:
        print("======= all =======")
        x = data[:, 0]
        y = data[:, 1]
    else:
        x = data[:, time, 0]
        y = data[:, time, 1]
    color = ['r', 'g', 'b', 'purple', 'orange', 'mediumturquoise', 'orange']
    label = ['sports', 'news', 'vehicles', 'food', 'cooking']
    #label = ['sports', 'news', 'vehicles', 'tarvel', 'kids', 'food', 'cooking']

    if isinstance(x, list):
        index = 1
        for item1, item2 in zip(x, y):
            print(item1.shape, item2.shape)
            print(item1[0], item2[0])
            ax = plt.subplot(1, 2, index)
            index += 1

            for j, s in enumerate(specific):
                idx = []
                for i in range(item1.shape[0]):
                    tmp = i // n_frames if time == -2 else i
                    category = info[str(tmp + start)]
                    #category = int(info['video' + str(i + start)])
                    value = category * 0.05
                    if s != -1 and category != s:
                        continue
                    idx.append(i)
                ax.scatter(item1[idx], item2[idx], c=color[j], label=label[j])
            ax.legend()
    else:
        for j, s in enumerate(specific):
            idx = []
            for i in range(x.shape[0]):
                tmp = i // n_frames if time == -2 else i
                category = info[str(tmp + start)]
                #category = int(info['video' + str(i + start)])
                value = category * 0.05
                if s != -1 and category != s:
                    continue
                idx.append(i)
            plt.scatter(x[idx], y[idx], c=color[j], label=label[j])

        cal_dis(x, y, all=(time==-2), n_frames=n_frames)
        
    plt.legend()
    plt.show()

def visualize(opt, option):
    '''
    plot(opt.em, option['info_json'], opt.pca_path, '%s_%d_%d.npy' % (opt.pca_name, opt.length_id, opt.gate_id), 
        -1 if opt.mean else opt.t,
        opt.specific
        )
    '''
    plot_several_category(opt.em, option['info_json'], opt.pca_path, '%s_%d_%d.npy' % (opt.pca_name, opt.length_id, opt.gate_id), 
        -1 if opt.mean else (-2 if opt.all else opt.t)
        )

def cal_centers(mode, outputfile, data, jsonfile, num_category=20):
    
    info = json.load(open(jsonfile))['id_to_category']
    #info = json.load(open("/home/yangbang/VideoCaptioning/MSRVTT/hidden_topic_mining/tc10s10/20_15/annotation.json"))['id_to_category']
    ind = [[] for _ in range(num_category)]
    

    start = 6513 if mode == 'validate' else 7010

    for i in range(data.shape[0]):
        category = info[str(i + start)]
        #category = int(info['video' + str(i + start)])
        ind[category].append(i)


    file = open(outputfile, 'w')
    n_frames = data.shape[1]
    assert len(data.shape) == 3
    header = ['time']
    for i in range(num_category):
        header.append('%d' % i)
    file.write(','.join(header) + '\n')



    inner_dis = []
    inter_dis = []
    for nf in range(n_frames):
        d = data[:, nf, :]
        inner = []
        centers = []
        for i in range(num_category):
            cluster = d[ind[i]]
            center = cluster.mean(0).unsqueeze(0)
            centers.append(center)
            ave_dis = F.pairwise_distance(center, cluster, p=2).mean(0)
            inner.append(ave_dis.view(1))
        inner_dis.append(torch.cat(inner, dim=0))

        inter = []
        for i in range(num_category):
            this_center = centers[i]
            other_centers = []
            for j in range(num_category):
                if j == i: continue
                other_centers.append(centers[j])
            other_centers = torch.cat(other_centers, dim=0)
            ave_dis = F.pairwise_distance(this_center, other_centers, p=2).mean(0)
            inter.append(ave_dis.view(1))
        inter_dis.append(torch.cat(inter, dim=0))

    inner_dis = torch.stack(inner_dis, dim=0)
    inter_dis = torch.stack(inter_dis, dim=0)

    for t in range(n_frames):
        write_data = ['%d' % t]
        for i in range(num_category):
            write_data.append('%.3f' % inner_dis[t, i])
        file.write(','.join(write_data) + '\n')
    
    for t in range(n_frames):
        write_data = ['%d' % t]
        for i in range(num_category):
            write_data.append('%.3f' % inter_dis[t, i])
        file.write(','.join(write_data) + '\n')

    file.close()

def main(opt):
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')
    opt_pth = os.path.join(opt.model_path, 'opt_info.json')
    option = json.load(open(opt_pth, 'r'))
    option.update(vars(opt))
    #print(option)

    model = get_model(option)
    checkpoint = torch.load(os.path.join(opt.model_path, 'best', opt.model_name))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)


    loader = get_loader(option, mode=opt.em, print_info=False, specific=opt.specific)
    vocab = loader.dataset.get_vocab()

    # rec length predicted results
    rec = {}
    length = len(option['modality']) - sum(option['skip_info'])
    num_gates = 3
    gate_data = [[[] for _ in range(num_gates)] for __ in range(length)]

    opt.pca_name = opt.pca_name + '_%s' % opt.em + ('_mean' if opt.mean else '') + ('_all' if opt.all else '')
    opt.pca_path = os.path.join(opt.pca_path, opt.model_path.split('/')[-2])
    print(opt.pca_path)
    print(opt.model_path.split('/'))
    if not os.path.exists(opt.pca_path):
        os.makedirs(opt.pca_path)

    if opt.plot:
        visualize(opt, option)
    elif opt.oe:
        metric = run_eval(option, model, None, loader, vocab, device, 
                        json_path=opt.json_path, json_name=opt.json_name, print_sent=opt.print_sent, no_score=opt.ns, save_videodatainfo=opt.sv)
        print(metric)
    else:
        for data in tqdm(loader, ncols=150, leave=False):
            with torch.no_grad():
                results, _, _ = get_forword_results(option, model, data, device=device, only_data=True)
                gate = results['gate']
                assert len(gate) == length
                assert len(gate[0]) == num_gates - 1
                assert isinstance(results['enc_output'], list)
                assert len(results['enc_output']) == length
                for i in range(length):
                    gate[i].append(results['enc_output'][i])
                    for j in range(num_gates):
                        gate_data[i][j].append(gate[i][j])

        for i in range(length):
            for j in range(num_gates):
                gate_data[i][j] = torch.cat(gate_data[i][j], dim=0) #[len_dataset, n_frames, dim_hidden]
                print(i, j, gate_data[i][j][0, 0, :10].tolist())
                
                if i == 0:
                    data = torch.cat([gate_data[0][j], torch.cat(gate_data[1][j], dim=0)], dim=0).cpu().numpy()
                else:
                    data = gate_data[i][j].cpu().numpy()
                name = '%s_%d_%d.npy' % (opt.pca_name, i, j)
                if opt.mean:
                    data = data.mean(1)
                    pca = manifold.TSNE(n_components=opt.pca)
                    #pca = PCA(n_components=opt.pca)     #加载PCA算法，设置降维后主成分数目为2
                    #collect = pca.fit_transform(data) #对样本进行降维
                    #print(pca.explained_variance_ratio_)
                    collect = pca.fit_transform(data) #对样本进行降维
                elif opt.all:
                    bsz, seq_len, dim = data.shape
                    data = data.reshape(bsz * seq_len, dim)
                    pca = manifold.TSNE(n_components=opt.pca)
                    collect = pca.fit_transform(data) #对样本进行降维
                else:
                    assert len(data.shape) == 3
                    seq_len = data.shape[1]
                    collect = []
                    for nf in range(seq_len):
                        x = data[:, nf, :]
                        pca = manifold.TSNE(n_components=opt.pca)
                        #pca = PCA(n_components=opt.pca)     #加载PCA算法，设置降维后主成分数目为2
                        reduced_x = pca.fit_transform(x) #对样本进行降维
                        collect.append(reduced_x)
                    collect = np.stack(collect, 1)
                print(name, collect.shape)
                np.save(os.path.join(opt.pca_path, name), collect)

            #print('--------')
            #print(i, resetgate[i].max(2)[0].max(0)[0].tolist())
            #print(i, inputgate[i].max(2)[0].max(0)[0].tolist())
            #print('--------')
            #print(i, resetgate[i].min(2)[0].min(0)[0].tolist())
            #print(i, inputgate[i].min(2)[0].min(0)[0].tolist())


        '''
        metric = run_eval(option, model, None, loader, vocab, device, 
                        json_path=opt.json_path, json_name=opt.json_name, print_sent=opt.print_sent, no_score=opt.ns, save_videodatainfo=opt.sv)
        print(metric)
        '''


def test(opt):
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')
    opt_pth = os.path.join(opt.model_path, 'opt_info.json')
    option = json.load(open(opt_pth, 'r'))
    option.update(vars(opt))

    model = get_model(option)
    checkpoint = torch.load(os.path.join(opt.model_path, 'best', opt.model_name))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)


    loader = get_loader(option, mode=opt.em, print_info=False, specific=opt.specific)
    vocab = loader.dataset.get_vocab()

    # rec length predicted results
    rec = {}
    length = len(option['modality']) - sum(option['skip_info'])
    num_gates = 3
    gate_data = [[[] for _ in range(num_gates)] for __ in range(length)]

    opt.pca_name = opt.pca_name + '_%s' % opt.em
    if not os.path.exists(opt.pca_path):
        os.makedirs(opt.pca_path)

    for data in tqdm(loader, ncols=150, leave=False):
        with torch.no_grad():
            results, _, _ = get_forword_results(option, model, data, device=device, only_data=True)
            gate = results['gate']
            assert len(gate) == length
            assert len(gate[0]) == num_gates
            for i in range(length):
                for j in range(num_gates):
                    gate_data[i][j].append(gate[i][j])

    for i in range(length):
        for j in range(num_gates):
            gate_data[i][j] = torch.cat(gate_data[i][j], dim=0) #[len_dataset, n_frames, dim_hidden]
            data = gate_data[i][j]
            name = '%s_%d_%d.csv' % (opt.pca_name, i, j)
            outputfile = os.path.join(opt.pca_path, name)
            cal_centers(opt.em, outputfile, data, option['info_json'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model_path', type=str, default="/home/yangbang/VideoCaptioning/0105save/MSRVTT/GRU_LSTM/ADD0_WA0_EBN1_SS1_WC0_Mi-Im_type2/")

    parser.add_argument('-model_name', default='0028_175313_179067_182774_182189_180426.pth.tar', type=str)
    #"/home/yangbang/VideoCaptioning/0105save/Youtube2Text/GRU_LSTM/ADD0_WA0_EBN1_SS1_WC0_Mi-Im_type2/best/0047_248567_257792_258102_257767_255302.pth.tar"
    #"/home/yangbang/VideoCaptioning/0105save/MSRVTT/GRU_LSTM/ADD0_WA0_EBN1_SS1_WC0_Mi-Im_type2/best/0028_175313_179067_182774_182189_180426.pth.tar"
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

    parser.add_argument('-oe', default=False,action='store_true')
    parser.add_argument('-dfi', '--dummy_feats_i', default=False, action='store_true')
    parser.add_argument('-dfm', '--dummy_feats_m', default=False, action='store_true')
    parser.add_argument('-dfa', '--dummy_feats_a', default=False, action='store_true')
    opt = parser.parse_args()

    main(opt)
    #test(opt)
