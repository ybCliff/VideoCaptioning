import sys
sys.path.append("..")
sys.path.append(".")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import copy
import math
import numpy as np
import json
from misc.optim import get_optimizer
from misc.crit import get_criterion
from misc.run import save_checkpoint
from misc.logger import CsvLogger
import h5py,random
from tqdm import tqdm

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
        self.random_type = opt.get('random_type', 'segment_random')
        assert self.mode in ['train', 'validate', 'test', 'all']
        assert self.random_type in ['segment_random', 'all_random']

        info = json.load(open(opt["info_json"]))
        self.splits = info['videos']
        self.tag_info = info['tag_info']

        self.specific = specific
        self.n_frames = opt['n_frames']
        self.equally_sampling = opt.get('equally_sampling', False)
        self.total_frames_length = opt.get('total_frames_length', 60)

        self.data_m = [self.load_database(opt["feats_m"]), opt["dim_m"], opt.get("dummy_feats_m", False)]

        self.all_caps_a_round = False
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
            tag = self.tag_info[vid]

            infoset.append({
                'vid': vid,
                'tag': tag
                })

        return infoset

    def shuffle(self):
        random.shuffle(self.infoset)


    def __getitem__(self, ix):
        vid = self.infoset[ix]['vid']
        tag = self.infoset[ix]['tag']

        frames_idx = get_frames_idx(
            self.total_frames_length, 
            self.n_frames, 
            self.random_type, 
            equally_sampling = True if self.mode != 'train' else self.equally_sampling)

        feats_m = self.load_feats(self.data_m, vid, frames_idx)

        return torch.FloatTensor(feats_m), torch.FloatTensor(tag)

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

    def load_feats(self, data, vid, frames_idx):
        databases, dim, dummy = data
        if not len(databases) or dummy:
            return np.zeros((self.n_frames, dim))

        feats = []
        for database in databases:
            if vid not in database.keys():
                return np.zeros((self.n_frames, dim))
            else:
                data = np.asarray(database[vid])
                if len(data.shape) == 1:
                    data = data[np.newaxis, :].repeat(self.total_frames_length, axis=0)
            feats.append(data)

        feats = np.concatenate(feats, axis=1)
        return feats[frames_idx]

def get_loader(opt, mode):
    dataset = VideoDataset(opt, mode)
    return DataLoader(
        dataset, 
        batch_size=opt["batch_size"], 
        shuffle=False
        )

class Model(nn.Module):
    def __init__(self, seq_len, dim_feats, dim_hidden, num_class, dropout_ratio, VIP_level=3, weighted_addition=False):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(dropout_ratio)
        self.rnn = nn.GRU(dim_feats, dim_hidden)
        self.bn = nn.BatchNorm1d(dim_hidden)
        self.vip = VIP_layer(
                seq_len=seq_len, 
                dim_feats=dim_hidden, 
                num_class=num_class, 
                dropout_ratio=dropout_ratio, 
                VIP_level=VIP_level
            )

    def forward(self, x):
        eo, eh = self.rnn(self.dropout(x))
        batch_size, seq_len, _ = eo.shape    
        eo = self.bn(eo.contiguous().view(batch_size * seq_len, -1)).view(batch_size, seq_len, -1)
        #print(eo.shape)
        return self.vip(eo)
        

class VIP_layer(nn.Module):
    def __init__(self, seq_len, dim_feats, num_class, dropout_ratio, VIP_level=3, weighted_addition=False):
        super(VIP_layer, self).__init__()
        
        VIP_level_limit = math.log(seq_len, 2)
        assert int(VIP_level_limit) == VIP_level_limit, 'seq_len must be a exponent of 2'
        assert VIP_level <= VIP_level_limit + 1, \
        'The maximun VIP_level for seq_len({}) is {:d}, {} is not allowed'.format(seq_len, VIP_level_limit+1, VIP_level)

        kernel_size = [(seq_len//(2**n), 1, 1) for n in range(VIP_level)]
        dilation = [(2**n, 1, 1) for n in range(VIP_level)]

        # Specific-timescale Pooling
        self.pooling = nn.ModuleList(
                [copy.deepcopy(
                    nn.MaxPool3d(
                        kernel_size=ks,
                        stride=1,
                        padding=0,
                        dilation=di
                    )
                ) for ks, di in zip(kernel_size, dilation)]
            )

        # corresponding dropout
        self.dropout = nn.ModuleList(
                [copy.deepcopy(nn.Dropout(p=dropout_ratio)) for _ in range(VIP_level)]
            )

        # Various-timescale Inference
        self.inference = nn.ModuleList(
                [copy.deepcopy(nn.Linear(dim_feats, num_class)) for _ in range(VIP_level)]
            )

        self.seq_len = seq_len
        self.dim_feats = dim_feats
        self.VIP_level = VIP_level
        self.weights = [(2**n if weighted_addition else 1) for n in range(VIP_level)] 

        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.001)
                nn.init.constant_(m.bias.data, 0)
        '''

    def forward(self, x):
        '''
            input: 
                -- x: [batch_size, seq_len, dim_feats]
            output:
                -- the prediction results: [batch_size, num_class]
        '''   
        x = x.permute(0,2,1).unsqueeze(3).unsqueeze(4) #[batch_size, dim_feats, D=seq_len, H=1, W=1]  
        collections = []
        for n in range(self.VIP_level):
            y = self.pooling[n](x).mean(2) #[batch_size, dim_feats, H=1, W=1]
            y = self.dropout[n](y).view(-1, self.dim_feats) #[batch_size, dim_feats]
            y = self.inference[n](y) #[batch_size, num_class]
            collections.append(self.weights[n] * y)

        results = torch.stack(collections, dim=0).sum(0) # w1 * y1 + w2 * y2 + ...
        return F.sigmoid(results)


def run_train(opt, model, crit, optimizer, loader, device, logger=None):
    model.train()
    crit.reset_loss_recorder()
    for feats, tag in tqdm(loader, ncols=150, leave=False):
        optimizer.zero_grad()

        feats = feats.to(device)
        tag = tag.to(device)
        results = {'pred_tag': model(feats), 'tag': tag}
        loss = crit.get_loss(results)
        loss.backward()
        optimizer.step()

    name, loss_info = crit.get_loss_info()
    if logger is not None:
        logger.write_text('\t'.join(['%10s: %05.3f' % (item[0], item[1]) for item in zip(name, loss_info)]))
    return loss_info[0]

def run_eval(opt, model, crit, loader, device):
    model.eval()
    crit.reset_loss_recorder()
    for feats, tag in tqdm(loader, ncols=150, leave=False):
        with torch.no_grad():
            feats = feats.to(device)
            tag = tag.to(device)
            results = {'pred_tag': model(feats), 'tag': tag}
            loss = crit.get_loss(results)

    name, loss_info = crit.get_loss_info()

    return loss_info[0]

if __name__ == '__main__':
    opt = {
        'dim_m': 2048,
        'dim_hidden': 512,
        'feats_m': ["/home/yangbang/VideoCaptioning/MSRVTT/feats/msrvtt_kinetics_60.hdf5"],
        'n_frames': 8,
        'num_class': 400,
        'random_type': 'segment_random',
        'equally_sampling': False,
        'info_json': "/home/yangbang/VideoCaptioning/MSRVTT/info_pad_mask_2.json",
        'VIP_level': 3,
        'dropout_ratio': 0.5,

        'optim': 'adam',
        'weight_decay': 5e-4,
        'alr': 2e-4,
        'amlr': 1e-4,
        'decay': 0.99,
        'epochs': 500,
        'batch_size': 64,

        'crit': ['obj'],
        'crit_name': ['Tag Loss'],
        'crit_scale': [1.0],
        'crit_key': [('pred_tag', 'tag')],

        'checkpoint_path': "/home/yangbang/VideoCaptioning/ARVC/tagging/VIP"
    }

    model = Model(
        seq_len=opt['n_frames'], 
        dim_feats=opt['dim_m'], 
        dim_hidden=opt['dim_hidden'],
        num_class=opt['num_class'], 
        dropout_ratio=opt['dropout_ratio'], 
        VIP_level=opt['VIP_level']
        )
    
    print(model)

    device = torch.device('cuda')
    model.to(device)
    optimizer = get_optimizer(opt, model)
    crit = get_criterion(opt)

    train_loader = get_loader(opt, 'train')
    vali_loader = get_loader(opt, 'validate')
    test_loader = get_loader(opt, 'test')
    logger = CsvLogger(
        filepath=opt["checkpoint_path"], 
        filename='trainning_record.csv', 
        fieldsnames=['epoch', 'train_loss', 'val_loss']
        )

    best = 1e6
    for epoch in range(opt['epochs']):
        train_loader.dataset.shuffle()
        logger.write_text("epoch %d lr=%g" % (epoch, optimizer.get_lr()))

        train_loss = run_train(opt, model, crit, optimizer, train_loader, device, logger=logger)
        optimizer.epoch_update_learning_rate()

        vali_loss = run_eval(opt, model, crit, vali_loader, device)
        
        save_checkpoint(
                {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'validate_result': vali_loss}, 
                vali_loss < best, 
                filepath=opt["checkpoint_path"], 
                filename='checkpoint.pth.tar'
            )

        logger.write_text('Epoch %4d: %05.3f (%05.3f)' % (epoch, vali_loss, vali_loss-best))
        if vali_loss < best:
            best = vali_loss

        logger.write({'epoch': epoch, 'train_loss': train_loss, 'val_loss': vali_loss})


    checkpoint = torch.load(opt["checkpoint_path"])
    model.load_state_dict(checkpoint['state_dict'])
    vali_loss = run_eval(opt, model, crit, vali_loader, device)
    test_loss = run_eval(opt, model, crit, test_loader, device)
    print(vali_loss, test_loss)


    