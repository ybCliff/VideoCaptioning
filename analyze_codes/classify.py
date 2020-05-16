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
import torch.nn.functional as F
import pickle
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import shutil

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
    def __init__(self, opt, mode, specific=-1):
        super(VideoDataset, self).__init__()
        self.mode = mode
        assert self.mode in ['train', 'validate', 'test', 'all']

        data = pickle.load(open(opt['info_corpus'], 'rb'))
        info = data['info']
        
        self.itoc = info['itoc']
        self.length_info = info['length_info']
        self.splits = info['split']
        self.split_category = info.get('split_category', None)

        self.specific = specific
        self.n_frames = opt['n_frames']

        self.data_i = [self.load_database(opt["feats_i"]), opt["dim_i"]]
        self.data_m = [self.load_database(opt["feats_m"]), opt["dim_m"]]
        self.data_e = [self.load_database(opt["feats_e"]), opt["dim_e"]]

        self.infoset = self.make_infoset()

    def make_infoset(self):
        infoset = []

        # decide the size of infoset
        if self.specific != -1:
            # we only evaluate partial examples with a specific category (MSRVTT, [0, 19])
            ix_set = [int(item) for item in self.split_category[self.mode][str(self.specific)]]
        else:
            # we evaluate all examples
            ix_set = [int(item) for item in self.splits[self.mode]]

        for ix in ix_set:
            vid = 'video%d' % ix
            category = self.itoc[ix]
            item = {
                'vid': vid,
                'category': category,
                }
            infoset.append(item)
        return infoset

    def __getitem__(self, ix):
        vid = self.infoset[ix]['vid']
        category = self.infoset[ix]['category']

        feats_i = self.load_feats_padding(self.data_i, vid)
        feats_m = self.load_feats_padding(self.data_m, vid)
        one_hot_embs = self.load_feats_padding(self.data_e, vid, padding=False)

        data = {}
        feats = []
        if feats_i is not None: 
            feats.append(torch.FloatTensor(feats_i).mean(0))
        if feats_m is not None: 
            feats.append(torch.FloatTensor(feats_m).mean(0))
        if len(feats):
            data['feats'] = torch.cat(feats, dim=0)

        if one_hot_embs is not None:
            data['embs'] = torch.FloatTensor(one_hot_embs)

        category = torch.LongTensor([category])

        return vid, data, category

    def __len__(self):
        return len(self.infoset)

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

    def load_feats_padding(self, data, vid, dummy=None, padding=True, scale=1):
        databases, dim = data
        if not len(databases):
            return None

        feats = []
        for database in databases:
            if vid not in database.keys():
                return np.zeros((self.n_frames, dim))
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
                    None, 
                    equally_sampling = True)
        else:
            frames_idx = resampling(source_length, self.n_frames)

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

def get_loader(opt, mode, specific=-1):
    dataset = VideoDataset(opt, mode, specific=specific)
    return DataLoader(
        dataset, 
        batch_size=opt["batch_size"], 
        shuffle=True if mode=='train' else False
        )

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, multiply=True):
        self.val = val
        if multiply:
            self.sum += val * n
        else:
            self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class Logger:
    def __init__(self, filepath='./'):
        self.log_path = filepath
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    def write_text(self, text, print_t=True):
        with open(os.path.join(self.log_path, 'log.txt'), 'a') as f:
            f.write('{}\n'.format(text))
        if print_t:
            tqdm.write(text)


def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar', best_model_name='model_best.pth.tar'):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    save_path = os.path.join(filepath, filename) 
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(filepath, best_model_name)
        shutil.copyfile(save_path, best_path)

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_category=20, dropout=0.5):
        super(Classifier, self).__init__()
        self.num = len(input_size) if isinstance(input_size, list) else 0
        assert self.num in [0, 1, 2]

        if self.num:
            self.net = []
            for i in range(self.num):
                tmp = nn.Sequential(
                        nn.Linear(input_size[i], hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size, num_category),
                    )
                self.add_module('net%d'%i, tmp)
                self.net.append(tmp)
        else:
            self.net = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size, num_category),
                    )

    def forward(self, inputs):
        if self.num:
            out = []
            for i in range(self.num):
                out.append(F.log_softmax(self.net[i](inputs[i]), dim=-1))
            return out

            return torch.stack(out, dim=0).mean(0)

        out = self.net(inputs)
        return F.log_softmax(out, dim=-1)

def training(model, loader, crit, optimizer, device):
    model.train()
    meter = AverageMeter()
    for vid, data, category in tqdm(loader, ncols=150, leave=False):
        optimizer.zero_grad()
        category = category.to(device).view(-1)
        feats = data.get('feats', None)
        embs = data.get('embs', None)

        inputs = []
        if feats is not None: inputs.append(feats.to(device))
        if embs is not None: inputs.append(embs.to(device))
        logits = model(inputs)
        loss = []
        for i in range(len(logits)):
            loss.append(crit(logits[i], category))
        loss = torch.stack(loss, dim=0).sum(0)

        meter.update(loss.item(), category.size(0))
        
        loss.backward()
        optimizer.step()

    return meter.avg

def evaluate(model, loader, crit, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for vid, data, category in tqdm(loader, ncols=150, leave=False):
        category = category.to(device).view(-1)
        feats = data.get('feats', None)
        embs = data.get('embs', None)

        inputs = []
        if feats is not None: inputs.append(feats.to(device))
        if embs is not None: inputs.append(embs.to(device))
        logits = model(inputs)
        loss = []
        res = []
        for i in range(len(logits)):
            loss.append(crit(logits[i], category))
            res.append(logits[i])
        loss = torch.stack(loss, dim=0).sum(0)
        logits = torch.stack(res, dim=0).mean(0)
        loss_meter.update(loss.item(), category.size(0))
        acc_meter.update((logits.max(-1)[1] == category).sum().item(), category.size(0), multiply=False)

    return loss_meter.avg, acc_meter.avg

def evaluate_ensemble(opt):
    e_pth = "./e_na_5all/"
    device = torch.device('cuda' if not opt.get('no_cuda', False) else 'cpu')
    
    opt['dim_e'] = 1
    opt['feats_e'] = e_pth + 'emb.hdf5'
    opt['modality'] = 'mie'
    loader = get_loader(opt, 'test')

    model_mi = torch.load("./mi/model_best.pth.tar")
    model_e = torch.load(e_pth + "model_best.pth.tar")

    model_mi = model_mi.to(device)
    model_e = model_e.to(device)
    model_mi.eval()
    model_e.eval()
    print(model_mi)
    print(model_e)

    acc_meter = AverageMeter()
    for vid, data, category in tqdm(loader, ncols=150, leave=False):
        category = category.to(device).view(-1)
        feats = data['feats'].to(device)
        embs = data['embs'].to(device)

        logits_mi = model_mi([feats])[0]
        logits_e = model_e([embs])[0]

        logits = logits_mi + logits_e
        #logits = logits_mi

        acc_meter.update((logits.max(-1)[1] == category).sum().item(), category.size(0), multiply=False)

    print("Ensemble Acc: %.2f" % (100*acc_meter.avg))



def get_one_hot_embeddings_from_captions(opt):
    import h5py
    import pickle
    vocab = {}
    vocab_file = opt["captions"].split('.')[0] + '_all_vocab_nv.txt'
    index = 0
    with open(vocab_file, 'r') as f:
        for line in f:
            word = line.strip()
            vocab[word] = index
            index += 1
    
    if not os.path.exists(opt['checkpoint_path']):
        os.makedirs(opt['checkpoint_path'])
    pth_to_save = os.path.join(opt["checkpoint_path"], 'emb.hdf5')
    data = h5py.File(pth_to_save, 'w')
    sents = pickle.load(open(opt["captions"], 'rb'))[0]
    for key in sents.keys():
        feat = np.zeros(index)
        words = ' '.join(sents[key]).split(' ')
        for w in words:
            if w in vocab.keys():
                feat[vocab[w]] = 1
        data[key] = feat

    return index, pth_to_save

if __name__ == '__main__':
    opt = {
        "info_corpus": "/home/yangbang/VideoCaptioning/MSRVTT/info_corpus_2.pkl",
        "n_frames": 8,
        "feats_i": "/home/yangbang/VideoCaptioning/MSRVTT/feats/msrvtt_resnet101_60.hdf5",
        "feats_m": "/home/yangbang/VideoCaptioning/MSRVTT/feats/msrvtt_kinetics_16_8.hdf5",
        "feats_e": "",

        "dim_i": 2048,
        "dim_m": 2048,
        "dim_e": 1,

        "batch_size": 128,
        "hidden_size": 512,
        "num_category": 20,
        "dropout": 0.5,
        "epochs": 100,
        "learning_rate": 0.0001,

        "checkpoint_path": './',
        #"captions": "/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/all/MSRVTT_nv_AEmp_i5b5a114_all.pkl",
        #"captions": "/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/all/MSRVTT_nv_mp_i5b5a114_all.pkl",
        "captions": "/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/all/MSRVTT_mp_mp_i5b5a114_all.pkl",
        'modality': 'e',
        
        'ensemble': False
    }
    set_seed()
    if opt['ensemble']:
        evaluate_ensemble(opt)
    else:
        opt['checkpoint_path'] += opt['modality']
        input_size = opt['dim_i'] + opt['dim_m']
        if 'm' not in opt['modality']:
            opt['feats_m'] = ""
            input_size -= opt['dim_m']
        if 'i' not in opt['modality']:
            opt['feats_i'] = ""
            input_size -= opt['dim_i']

        if 'e' in opt['modality']: 
            assert opt["captions"]
            #opt['checkpoint_path'] += '_na_AEmp_5all'
            #opt['checkpoint_path'] += '_na_mp_5all'
            opt['checkpoint_path'] += '_mp_mp_5all'

            #opt['checkpoint_path'] += '_namp_5all'
            #opt['checkpoint_path'] += '_mp_5all'
            #opt['dim_e'], opt['feats_e'] = get_one_hot_embeddings_from_captions(opt)

            opt['dim_e'] = 512
            opt['feats_e'] = "/home/yangbang/VideoCaptioning/ARVC/collect_embs/MSRVTT_nv_AEmp_i5b5a114_all.hdf5"
            opt['feats_e'] = "/home/yangbang/VideoCaptioning/ARVC/collect_embs/MSRVTT_nv_mp_i5b5a114_all.hdf5"
            opt['feats_e'] = "/home/yangbang/VideoCaptioning/ARVC/collect_embs/MSRVTT_mp_mp_i5b5a114_all.hdf5"
            input_size = [opt['dim_e']] if input_size == 0 else [input_size, opt['dim_e']]
        else:
            input_size = [input_size]



        model = Classifier(
                input_size = input_size, 
                hidden_size = opt['hidden_size'], 
                num_category = opt['num_category'], 
                dropout = opt['dropout']
            )
        print(model)
        print('| num. model params: {} (num. trained: {})'.format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        ))
        crit = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=opt["learning_rate"])
        device = torch.device('cuda' if not opt.get('no_cuda', False) else 'cpu')

        
        vali_loader = get_loader(opt, 'validate')
        test_loader = get_loader(opt, 'test')

        model = model.to(device)
        best_res = 0
        logger = Logger(opt['checkpoint_path'])

        for i in range(opt["epochs"]):
            train_loader = get_loader(opt, 'train')

            train_loss = training(model, train_loader, crit, optimizer, device)
            vali_loss, res = evaluate(model, vali_loader, crit, device)
            text = "Epoch %d\tTraining Loss %.2f\tValidation Loss %.2f\tAccuracy %.2f (%.2f)" % (i+1, train_loss, vali_loss, 100*res, 100*best_res)
            logger.write_text(text)

            save_checkpoint(
                    model,
                    res > best_res,
                    filepath = opt['checkpoint_path']
                )
            if res > best_res:
                best_res = res

        model = torch.load(os.path.join(opt['checkpoint_path'], 'model_best.pth.tar'))
        #model.load_state_dict(checkpoint['state_dict'])
        _, acc = evaluate(model, test_loader, crit, device)
        logger.write_text("Test Accuracy %.2f" % (100*acc))



    

