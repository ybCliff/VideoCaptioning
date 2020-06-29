''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import os,json
from misc.run import get_model, get_loader, run_eval, get_forword_results
from sklearn.decomposition import PCA
from sklearn import manifold
import numpy as np
import os

import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import h5py

def get_frames_idx(length, n_frames, random_type, equally_sampling=False):
    bound = [int(i) for i in np.linspace(0, length, n_frames+1)]
    idx = []
    all_idx = [i for i in range(length)]

    if random_type == 'all_random' and not equally_sampling:
        idx = random.sample(all_idx, n_frames)
    else:
        for i in range(n_frames):
            if not equally_sampling:
                tmp = np.random.randint(bound[i], bound[i+1])
            else:
                tmp = (bound[i] + bound[i+1]) // 2
            idx.append(tmp)

    return sorted(idx)

def load_feats(data, vid, frames_idx, n_frames=8, total_frames_length=60):
    databases, dim = data
    if not len(databases):
        return np.zeros((n_frames, dim))

    feats = []
    for database in databases:
        if vid not in database.keys():
            return np.zeros((n_frames, dim))
        else:
            data = np.asarray(database[vid])
            if len(data.shape) == 1:
                data = data[np.newaxis, :].repeat(total_frames_length, axis=0)
        feats.append(data)
        #print(data.shape)

    feats = np.concatenate(feats, axis=1)
    return feats[frames_idx]

def load_database(path):
    database = []
    if isinstance(path, list):
        for p in path:
            if '.hdf5' in p:
                database.append(h5py.File(p, 'r'))
    else:
        if '.hdf5' in path:
            database.append(h5py.File(path, 'r'))
    return database

def cal_centers(mode, data, jsonfile="/home/yangbang/VideoCaptioning/MSRVTT/info_pad_mask_2.json", num_category=20, n_frames=8, all=False):
    
    info = json.load(open(jsonfile))['id_to_category']
    #info = json.load(open("/home/yangbang/VideoCaptioning/MSRVTT/hidden_topic_mining/tc10s10/20_15/annotation.json"))['id_to_category']
    ind = [[] for _ in range(num_category)]
    

    start = 6513 if mode == 'validate' else 7010

    for i in range(data.shape[0]):
        category = info[str((i // n_frames if all else i) + start)]
        #category = int(info['video' + str(i + start)])
        ind[category].append(i)

    inner = []
    '''
    centers = []
    for i in range(num_category):
        cluster = data[ind[i]]
        center = cluster.mean(0).unsqueeze(0)
        centers.append(center)
        ave_dis = F.pairwise_distance(center, cluster, p=2).mean(0)
        inner.append(ave_dis.view(-1))
    '''
    for i in range(num_category):
        cluster = data[ind[i]]
        dis = []
        for j in range(len(ind[i]) - 1):
            this_point = cluster[j, :]
            others = cluster[j+1, :]
            if len(others.shape) == 1:
                others = others.unsqueeze(0)
            dis.append(F.pairwise_distance(this_point, others, p=2).view(-1))
        dis = torch.cat(dis, dim=0)#.mean(0)
        inner.append(dis.view(-1))
    inner_dis = torch.cat(inner, dim=0).mean(0)

    inter = []
    for i in range(num_category-1):
        this_cluster = data[ind[i]]
        for j in range(i+1, num_category):
            other_cluster = data[ind[j]]

            item_wise = []
            for k in range(this_cluster.size(0)):
                ave_dis = F.pairwise_distance(this_cluster[k], other_cluster, p=2)
                item_wise.append(ave_dis.view(-1))
            ij_dis = torch.cat(item_wise, dim=0)#.mean(0)

            #print(i, j, '%.3f' % ij_dis)
            inter.append(ij_dis.view(-1))
    inter_dis = torch.cat(inter, dim=0).mean(0)

    return inner_dis, inter_dis



def plot_several_category(mode, jsonfile, path, time, 
    specific=[
        3, 
        4, 
        9, 
        #11, 
        #14, 
        16, 
        17
        ], n_frames=8):
    
    def cal_dis(x, y, start, num_category=20, all=False, n_frames=8):
        print(start, all)
        ave_dis = []
        centers = []
        all_idx = []
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
            centers.append(center)
            ave_dis.append(F.pairwise_distance(center, cluster, p=2).view(-1))
            all_idx.append(idx)

        intra_dis = torch.cat(ave_dis, 0).mean(0)
        '''
        inter = []
        for i in range(num_category-1):
            this_cluster = torch.stack([torch.from_numpy(x[all_idx[i]]), torch.from_numpy(y[all_idx[i]])], dim=1)

            #other_idx = all_idx[i+1].copy()
            #for j in range(i+2, )
            for j in range(i+1, num_category):
                other_cluster = torch.stack([torch.from_numpy(x[all_idx[j]]), torch.from_numpy(y[all_idx[j]])], dim=1)

                item_wise = []
                for k in range(this_cluster.size(0)):
                    ave_dis = F.pairwise_distance(this_cluster[k], other_cluster, p=2)
                    item_wise.append(ave_dis.view(-1))
                ij_dis = torch.cat(item_wise, dim=0).mean(0)

                print(i, j, '%.3f' % ij_dis)
                inter.append(ij_dis.view(-1))
        inter_dis = torch.cat(inter, dim=0).mean(0)
        print(len(inter))
        '''
        inter = []
        for i in range(num_category-1):
            this_center = centers[i]
            other_centers = []
            for j in range(i+1, num_category):
                other_centers.append(centers[j])
            other_centers = torch.cat(other_centers, dim=0)
            ave_dis = F.pairwise_distance(this_center, other_centers, p=2)
            inter.append(ave_dis.view(-1))
        inter_dis = torch.cat(inter, dim=0).mean(0)
        print(len(inter))
        
        return intra_dis, inter_dis

    info = json.load(open(jsonfile))['id_to_category']

    for num, m in enumerate(['i', 'm', 'a', 'im', 'ia', 'ma', 'ima']):
        ax = plt.subplot(3, 3, num + 1)

        if mode == 'validate':
            start = 6513
        else:
            start = 7010

        data = np.load(os.path.join(path, '%s.npy' % m))
        if time == -1:
            print("======= mean =======")
            x = data[:, 0]
            y = data[:, 1]
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
            ax.scatter(x[idx], y[idx], c=color[j], label=label[j])

        intra, inter = cal_dis(x, y, start, all=(time==-2), n_frames=n_frames)
        #intra, inter = cal_centers(mode, torch.stack([torch.from_numpy(x), torch.from_numpy(y)], dim=1).cuda(), jsonfile, all=(time==-2))
        #ax.set_title('%s %.2f %.2f' % (m, intra, inter))
        ax.set_title('%s %.2f' % (m, intra))
        #ax.legend()
    #plt.legend()
    plt.show()

def visualize(opt):
    plot_several_category(opt['em'], "/home/yangbang/VideoCaptioning/MSRVTT/info_pad_mask_2.json", opt['pca_path'], 
        -1 if opt['mean'] else (-2 if opt['all'] else opt['t'])
        )


def main(opt):
    opt.update({
        'feats_i': "/home/yangbang/VideoCaptioning/MSRVTT/feats/msrvtt_R101.hdf5",
        'feats_m': "/home/yangbang/VideoCaptioning/MSRVTT/feats/msrvtt_c3d_60_fc6.hdf5", #"/home/yangbang/VideoCaptioning/MSRVTT/feats/msrvtt_kinetics_60.hdf5",
        'feats_a': ["/home/yangbang/VideoCaptioning/MSRVTT/feats/msrvtt_vggish_60.hdf5", "/home/yangbang/VideoCaptioning/MSRVTT/feats/fvdb_260.hdf5", "/home/yangbang/VideoCaptioning/MSRVTT/feats/vtt_boaw256.hdf5"],
        'dim_i': 2048,
        'dim_m': 4096,
        'dim_a': 644
    })
    data_i = [load_database(opt["feats_i"]), opt["dim_i"]]
    data_m = [load_database(opt["feats_m"]), opt["dim_m"]]
    data_a = [load_database(opt["feats_a"]), opt["dim_a"]]


    length, n_frames, random_type, equally_sampling = 60, 8, None, True
    frames_idx = get_frames_idx(length, n_frames, random_type, equally_sampling=equally_sampling)

    if opt['em'] == 'validate':
        begin, end = 6513, 7010 
    elif opt['em'] == 'test':
        begin, end = 7010, 10000
    else:
        begin, end = 0, 6513

    feats_i, feats_m, feats_a = [], [], []
    for ix in range(begin, end):
        vid = 'video%d' % ix
        i = load_feats(data_i, vid, frames_idx)
        m = load_feats(data_m, vid, frames_idx)
        a = load_feats(data_a, vid, frames_idx)

        feats_i.append(i)
        feats_m.append(m)
        feats_a.append(a)

    feats_i = np.array(feats_i)
    feats_m = np.array(feats_m)
    feats_a = np.array(feats_a)

    mapping = {
        'a': feats_a,
        'm': feats_m,
        'i': feats_i
    }

    if opt['plot']:
        visualize(opt)
    elif opt['cal']:
        for modality in ['i', 'm', 'a', 'im', 'ia', 'ma', 'ima']:
            feats = []
            for char in modality:
                feats.append(mapping[char])

            data = np.concatenate(feats, axis=2)
            data = data.mean(1)
            intra, inter = cal_centers(opt['em'], torch.from_numpy(data).cuda())
            print('%4s\tIntra: %05.3f\tInter: %05.3f' % (modality, intra, inter))


    else:
        for modality in ['i', 'm', 'a', 'im', 'ia', 'ma', 'ima']:
            feats = []
            for char in modality:
                feats.append(mapping[char])

            data = np.concatenate(feats, axis=2)
            name = '%s.npy' % modality
            
            if opt['mean']:
                data = data.mean(1)
                pca = manifold.TSNE(n_components=2)
                collect = pca.fit_transform(data) #对样本进行降维
            elif opt['all']:
                bsz, seq_len, dim = data.shape
                data = data.reshape(bsz * seq_len, dim)
                pca = manifold.TSNE(n_components=2)
                collect = pca.fit_transform(data) #对样本进行降维
            else:
                assert len(data.shape) == 3
                seq_len = data.shape[1]
                collect = []
                for nf in range(seq_len):
                    x = data[:, nf, :]
                    pca = manifold.TSNE(n_components=2)
                    #pca = PCA(n_components=opt.pca)     #加载PCA算法，设置降维后主成分数目为2
                    reduced_x = pca.fit_transform(x) #对样本进行降维
                    collect.append(reduced_x)
                collect = np.stack(collect, 1)
            print(name, collect.shape)
            np.save(os.path.join(opt['pca_path'], name), collect)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-em', type=str, default='test')
    parser.add_argument('-specific', default=-1, type=int)
    parser.add_argument('--pca_path', default='./pca_results')
    parser.add_argument('-pn', '--pca_name', default='')
    parser.add_argument('-mean', default=False, action='store_true')
    parser.add_argument('-all', default=False, action='store_true')

    parser.add_argument('-plot', default=False, action='store_true')
    parser.add_argument('-cal', default=False, action='store_true')
    parser.add_argument('-t', default=0, type=int)
    opt = parser.parse_args()
    opt.pca_path = os.path.join(opt.pca_path, opt.em)
    tmp = 'mean' if opt.mean else ('all' if opt.all else '')
    if tmp:
        opt.pca_path = os.path.join(opt.pca_path, tmp)
    if not os.path.exists(opt.pca_path):
        os.makedirs(opt.pca_path)
    main(vars(opt))
    #test(opt)

# python feats_tsne.py -mean -plot