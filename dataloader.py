import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy
import math
import h5py
import models.Constants as Constants
from bisect import bisect_left
import torch.nn.functional as F
import pickle
from pandas.io.json import json_normalize

def resampling(source_length, target_length):
    return [round(i * (source_length-1) / (target_length-1)) for i in range(target_length)]

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

class VideoDataset(Dataset):
    def __init__(self, opt, mode, print_info=False, shuffle_feats=0, specific=-1):
        super(VideoDataset, self).__init__()
        self.mode = mode
        self.random_type = opt.get('random_type', 'segment_random')
        assert self.mode in ['train', 'validate', 'test', 'all', 'trainval']
        assert self.random_type in ['segment_random', 'all_random']

        # load the json file which contains information about the dataset
        data = pickle.load(open(opt['info_corpus'], 'rb'))
        info = data['info']
            
        self.itow = info['itow']
        self.wtoi = {v: k for k, v in self.itow.items()}
        self.itoc = info.get('itoc', None)        
        self.itop = info.get('itop', None)
        self.itoa = info.get('itoa', None) 
        self.length_info = info['length_info']
        self.splits = info['split']
        if self.mode == 'trainval':
            self.splits['trainval'] = self.splits['train'] + self.splits['validate']

        self.split_category = info.get('split_category', None)
        self.id_to_vid = info.get('id_to_vid', None)

        self.captions = data['captions']
        self.pos_tags = data['pos_tags']

        self.references = pickle.load(open(opt['reference'], 'rb'))
        
        self.specific = specific
        self.num_category = opt.get('num_category', 20)

        self.max_len = opt["max_len"]
        self.n_frames = opt['n_frames']
        self.equally_sampling = opt.get('equally_sampling', False)
        self.total_frames_length = opt.get('total_frames_length', 60)

        self.data_i = [self.load_database(opt["feats_i"]), opt["dim_i"], opt.get("dummy_feats_i", False)]
        self.data_m = [self.load_database(opt["feats_m"]), opt["dim_m"], opt.get("dummy_feats_m", False)]
        #self.data_i = [[], opt["dim_i"], opt.get("dummy_feats_i", False)]
        #self.data_m = [[], opt["dim_m"], opt.get("dummy_feats_m", False)]
        self.data_a = [self.load_database(opt["feats_a"]), opt["dim_a"], opt.get("dummy_feats_a", False)]
        

        self.data_s = [self.load_database(opt.get("feats_s", [])), opt.get("dim_s", 10), False]
        self.data_t = [self.load_database(opt.get("feats_t", [])), opt.get('dim_t', 10), False]

        self.mask_prob = opt.get('teacher_prob', 1)
        self.decoder_type = opt['decoder_type']
        self.random = np.random.RandomState(opt.get('seed', 0))

        self.obj = self.load_database(opt.get('object_path', ''))

        self.all_caps_a_round = opt['all_caps_a_round']
        

        self.load_feats_type = opt['load_feats_type']
        self.method = opt.get('method', 'mp')
        self.demand = opt['demand']

        self.opt = opt
        if print_info: self.print_info(opt)
        
        self.beta_low, self.beta_high = opt.get('beta', [0, 1])
        if (opt.get('triplet', False) or opt.get('knowledge_distillation_with_bert', False)) and self.mode == 'train':
            self.bert_embeddings = self.load_database(opt['bert_embeddings'])
        else:
            self.bert_embeddings = None

        if opt.get('load_generated_captions', False):
            self.generated_captions = pickle.load(open(opt['generated_captions'], 'rb'))
            assert self.mode in ['test']
        else:
            self.generated_captions = None

        self.infoset = self.make_infoset()

    def get_references(self):
        return self.references

    def get_preprocessed_references(self):
        return self.captions

    def make_infoset(self):
        infoset = []

        # decide the size of infoset
        if self.specific != -1:
            # we only evaluate partial examples with a specific category (MSRVTT, [0, 19])
            ix_set = [int(item) for item in self.split_category[self.mode][self.specific]]
        else:
            # we evaluate all examples
            ix_set = [int(item) for item in self.splits[self.mode]]

        vatex = self.opt['dataset'] == 'VATEX' and self.mode == 'test'


        for ix in ix_set:
            vid = 'video%d' % ix
            if vatex:
                category = 0
                captions = [[0]]
                pos_tags = [[0]]
                length_target = [0]
            else:
                category = self.itoc[ix] if self.itoc is not None else 0
                captions = self.captions[vid]
                pos_tags = self.pos_tags[vid] if self.pos_tags is not None else ([None] * len(captions))

                # prepare length info for each video example, only if decoder_type == 'NARFormmer'
                # e.g., 'video1': [0, 0, 3, 5, 0]
                if self.length_info is None:
                    length_target = np.zeros(self.max_len)
                else:
                    length_target = self.length_info[vid]
                    #length_target = length_target[1:self.max_len+1]
                    length_target = length_target[:self.max_len]
                    if len(length_target) < self.max_len:
                        length_target += [0] * (self.max_len - len(length_target))

                    #right_sum = sum(length_target[self.max_len+1:])
                    #length_target[-1] += right_sum  
                    length_target = np.array(length_target) / sum(length_target)


            if self.mode == 'train' and self.all_caps_a_round:
                # infoset will contain all captions
                for i, (cap, pt) in enumerate(zip(captions, pos_tags)):
                    item = {
                            'vid': vid,
                            'labels': cap,
                            'pos_tags': pt,
                            'category': category,
                            'length_target': length_target,
                            'cap_id': i,
                            }

                    infoset.append(item)
            else:                
                if self.generated_captions is not None:
                    # edit the generated captions
                    cap = self.generated_captions[vid][-1]['caption']
                    #print(cap)
                    labels = [Constants.BOS]
                    for w in cap.split(' '):
                        labels.append(self.wtoi[w])
                    labels.append(Constants.EOS)
                    #print(labels)
                    item = {
                        'vid': vid,
                        'labels': labels,
                        'pos_tags': pos_tags[0],
                        'category': category,
                        'length_target': length_target
                        }
                else:
                    # infoset will contain partial captions, one caption per video clip
                    cap_ix = random.randint(0, len(self.captions[vid]) - 1) if self.mode == 'train' else 0
                    #print(captions[0])
                    item = {
                        'vid': vid,
                        'labels': captions[cap_ix],
                        'pos_tags': pos_tags[cap_ix],
                        'category': category,
                        'length_target': length_target,
                        'cap_id': cap_ix,
                        }
                infoset.append(item)
        return infoset

    def shuffle(self):
        random.shuffle(self.infoset)

    def __getitem__(self, ix):
        vid = self.infoset[ix]['vid']
        labels = self.infoset[ix]['labels']
        taggings = self.infoset[ix]['pos_tags']
        category = self.infoset[ix]['category']
        length_target = self.infoset[ix]['length_target']

        cap_id = self.infoset[ix].get('cap_id', None)
        if cap_id is not None and self.bert_embeddings is not None:
            bert_embs = np.asarray(self.bert_embeddings[0][vid])#[cap_id]
        else:
            bert_embs = None

        attribute = self.itoa[vid]


        frames_idx = get_frames_idx(
            self.total_frames_length, 
            self.n_frames, 
            self.random_type, 
            equally_sampling = True if self.mode != 'train' else self.equally_sampling
        ) if self.load_feats_type == 0 else None

        load_feats_func = self.load_feats if self.load_feats_type == 0 else self.load_feats_padding

        feats_i = load_feats_func(self.data_i, vid, frames_idx)
        feats_m = load_feats_func(self.data_m, vid, frames_idx, padding=False)#, scale=0.1)
        feats_a = load_feats_func(self.data_a, vid, frames_idx)#, padding=False)
        feats_s = load_feats_func(self.data_s, vid, frames_idx)

        feats_t = load_feats_func(self.data_t, vid, frames_idx)#, padding=False)

        results = self.make_source_target(labels, taggings)

        tokens, labels, pure_target, taggings = map(
            lambda x: results[x], 
            ["dec_source", "dec_target", "pure_target", "tagging"]
        )
        tokens_1 = results.get('dec_source_1', None)
        labels_1 = results.get('dec_target_1', None)
 
        data = {}
        data['feats_i'] = torch.FloatTensor(feats_i)
        data['feats_m'] = torch.FloatTensor(feats_m)#.mean(0).unsqueeze(0).repeat(self.n_frames, 1)
        data['feats_a'] = torch.FloatTensor(feats_a)
        data['feats_s'] = F.softmax(torch.FloatTensor(feats_s), dim=1)

        #print(feats_t.shape)
        data['feats_t'] = torch.FloatTensor(feats_t)

        data['tokens'] = torch.LongTensor(tokens)
        data['labels'] = torch.LongTensor(labels)
        data['pure_target'] = torch.LongTensor(pure_target)
        data['length_target'] = torch.FloatTensor(length_target)

        data['attribute'] = torch.FloatTensor(attribute)

        if tokens_1 is not None:
            data['tokens_1'] = torch.LongTensor(tokens_1)
            data['labels_1'] = torch.LongTensor(labels_1)

        if taggings is not None:
            data['taggings'] = torch.LongTensor(taggings)

        if bert_embs is not None:
            data['bert_embs'] = torch.FloatTensor(bert_embs)

        if self.decoder_type == 'LSTM' or self.decoder_type == 'ENSEMBLE':
            tmp = np.zeros(self.num_category)
            tmp[category] = 1
            data['category'] = torch.FloatTensor(tmp)
        else:
            data['category'] = torch.LongTensor([category])
        
        if frames_idx is not None:
            data['frames_idx'] = frames_idx
        data['video_ids'] = vid

        if len(self.obj):
            data['obj'] = torch.FloatTensor(np.asarray(self.obj[0][vid]))
        return data

    def __len__(self):
        return len(self.infoset)

    def get_mode(self):
        return self.id_to_vid, self.mode

    def set_splits_by_json_path(self, json_path):
        self.splits = json.load(open(json_path))['videos']

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.itow

    def print_info(self, opt):
        print('vocab size is ', len(self.itow))
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['validate']))
        print('number of test videos: ', len(self.splits['test']))
        print('load image feats  (%d) from %s' % (opt["dim_i"], opt["feats_i"]))
        print('load motion feats (%d) from %s' % (opt["dim_m"], opt["feats_m"]))
        print('load audio feats  (%d )from %s' % (opt["dim_a"], opt["feats_a"]))
        print('max sequence length in data is', self.max_len)
        print('load feats type: %d' % self.load_feats_type)

    def load_database(self, path):
        if not path:
            return []
        database = []
        if isinstance(path, list):
            for p in path:
                if '.hdf5' in p:
                    database.append(h5py.File(p, 'r'))
        else:
            if '.hdf5' in path:
                database.append(h5py.File(path, 'r'))
        return database

    def load_feats(self, data, vid, frames_idx, padding=True):
        databases, dim, dummy = data
        if not len(databases) or dummy:
            return np.zeros((self.n_frames, dim))

        feats = []
        for database in databases:
            if vid not in database.keys():
                return np.zeros((self.n_frames, dim))
            else:
                data = np.asarray(database[vid])
                if len(data.shape) == 1 and padding:
                    data = data[np.newaxis, :].repeat(self.total_frames_length, axis=0)
            feats.append(data)

        if len(feats[0].shape) == 1:
            feats = np.concatenate(feats, axis=0)
            return feats
        feats = np.concatenate(feats, axis=1)
        return feats[frames_idx]

    def load_feats_padding(self, data, vid, dummy=None, padding=True, scale=1):
        databases, dim, _ = data
        if not len(databases):
            return np.zeros((self.n_frames, dim))

        feats = []
        for database in databases:
            if vid not in database.keys():
                if padding:
                    return np.zeros((self.n_frames, dim))
                else:
                    return np.zeros(dim)
            else:
                data = np.asarray(database[vid])
                if len(data.shape) == 1 and padding:
                    data = data[np.newaxis, :].repeat(self.total_frames_length, axis=0)
            feats.append(data * scale)

        if len(feats[0].shape) == 1:
            feats = np.concatenate(feats, axis=0)
            return feats
        feats = np.concatenate(feats, axis=1)
        source_length = feats.shape[0]
        
        if source_length > self.n_frames:
            frames_idx = get_frames_idx(
                    source_length, 
                    self.n_frames, 
                    self.random_type, 
                    equally_sampling = True if self.mode != 'train' else self.equally_sampling)
        else:
            frames_idx = resampling(source_length, self.n_frames)
            #frames_idx = [i for i in range(feats.size(0))]
            #frames_idx += [-1] * (self.n_frames - feats.size(0))

        #print(vid, feats.sum(), feats.shape, frames_idx)
        return feats[frames_idx]

    def padding(self, seq, add_eos=True):
        if seq is None:
            return None
        res = seq.copy()
        if len(res) > self.max_len:
            res = res[:self.max_len]
            if add_eos:
                res[-1] = Constants.EOS
        else:
            res += [Constants.PAD] * (self.max_len - len(res))
        return res

    def make_source_target(self, target, tagging):
        if self.decoder_type == 'NARFormer':
            results = self.source_target_mlm(target[1:-1]) # exclude <bos> <eos>
        else:
            # ARFormer
            results = {
                'dec_source': self.padding(target, add_eos=True), 
                'dec_target': self.padding(target, add_eos=True)
            }

        assert len(results['dec_source']) == len(results['dec_target'])

        if self.method in ['ag', 'nv']:
            results.update(self.source_target_visual_word(target=target, pos_tag=tagging))

        if 'pure_target' not in results.keys():
            results['pure_target'] = self.padding(target.copy(), add_eos=True)
        if 'tagging' not in results.keys():
            results['tagging'] = self.padding(tagging, add_eos=True)

        return results

    def source_target_mlm(self, target):
        assert target[0] != Constants.BOS
        assert target[-1] != Constants.EOS

        min_num_masks = 1
        dec_source = torch.LongTensor(target)
        dec_target_cp = torch.LongTensor(target)
        dec_target = torch.LongTensor([Constants.PAD] * len(dec_source))

        if self.mode == 'train':
            if min_num_masks >= len(dec_source):
                ind = np.array([],dtype=np.uint8)
            else:
                low = max(int(len(dec_source) * self.beta_low), min_num_masks)
                high = max(int(len(dec_source) * self.beta_high), min_num_masks+1)
                sample_size = self.random.randint(low, high)
                ind = self.random.choice(len(dec_source) , size=sample_size, replace=False)
            
            dec_source[ind] = Constants.MASK
            dec_target[ind] = dec_target_cp[ind]
        else:
            dec_source[dec_source!=Constants.PAD] = Constants.MASK
            dec_target = dec_target_cp           

        dec_source = self.padding(dec_source.tolist(), add_eos=False)
        dec_target = self.padding(dec_target.tolist(), add_eos=False)
        pure_target = self.padding(target, add_eos=False)
        
        return {'dec_source': dec_source, 'dec_target': dec_target, 'pure_target': pure_target}

    def source_target_visual_word(self, **kwargs):
        target = kwargs['target']
        pos_tag = kwargs['pos_tag']
        sent_length = len(target[1:-1]) # exclude <bos> <eos>

        if self.decoder_type == 'NARFormer':
            visual_tag = Constants.BOS
            target_tag = Constants.MASK
        else:
            visual_tag = Constants.MASK
            target_tag = Constants.BOS

        if self.mode != 'train':
            dec_target_1 = [0]
            dec_source_1 = [0]
        else:
            assert len(target) == len(pos_tag)
            assert self.itop is not None

            dec_target_cp = torch.LongTensor(target[1:-1])

            dec_source_1 = self.padding([visual_tag] * sent_length if self.decoder_type == 'NARFormer' else len(target), 
                add_eos=False if self.decoder_type == 'NARFormer' else True)

            # get the position of tokens that have the pos_tag we demand
            pos_satisfied_ind = []
            for i, item in enumerate(pos_tag[1:-1]):
                w = self.itow[target[i+1]]
                # we ignore verb ``be''
                if self.itop[item] in self.demand and w not in ['is', 'are', 'was', 'were', 'be']:
                    pos_satisfied_ind.append(i)

            pos_satisfied_ind = np.array(pos_satisfied_ind)
            
            # decoder1 need to predict tokens with satisfied pos_tag from scratch
            # meanwhile, decoder1 should learn to keep the remaining tokens (i.e., <mask>) unchanged
            dec_target_1 = torch.LongTensor([target_tag] * sent_length)
            dec_target_1[pos_satisfied_ind] = dec_target_cp[pos_satisfied_ind]

            if self.decoder_type == 'NARFormer':
                dec_target_1 = self.padding(dec_target_1.tolist(), add_eos=False)
            else:
                # when training with autoregressive transformer, the first token will be ignored, i.e., label = dec_target_1[1:]
                dec_target_1 = self.padding([target[0]] + dec_target_1.tolist() + [Constants.BOS], add_eos=True)

        #print(dec_source_1, dec_target_1)
        return {'dec_source_1': dec_source_1, 'dec_target_1': dec_target_1}

