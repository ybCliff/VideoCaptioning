import json
import os
import numpy as np
import opts as opts
import torch
import torch.optim as optim
import misc.utils as utils
from misc.logger import CsvLogger
from models import get_model
from misc.run import train_network_all,train_beam_decoder

from torch.utils.data import DataLoader
from misc.cocoeval import suppress_stdout_stderr, COCOScorer
from pandas.io.json import json_normalize
import tqdm
import warnings
import shutil
from tqdm import tqdm
warnings.filterwarnings('ignore')
import random
import pickle

def get_dir(opt, key, mid_path='/', post='', pre=False, prefix=''):
    if not opt.get(key, ''):
        return ''
    if prefix:
        if isinstance(opt[key], list):
            for i in range(len(opt[key])):
                opt[key][i] = prefix + opt[key][i]
        else:
            opt[key] = prefix + opt[key]
    if pre:
        pre_str = 'msvd_' if opt['dataset'] == 'Youtube2Text' else ('msrvtt_' if opt['dataset'] == 'MSRVTT' else 'vatex_')
        if isinstance(opt[key], list):
            for i in range(len(opt[key])):
                opt[key][i] = pre_str + opt[key][i]
        else:
            opt[key] = pre_str + opt[key]
    if post:
        opt[key] += post
    res = []
    if isinstance(opt[key], list):
        if not opt[key][0]: return ''
        for i in range(len(opt[key])):
            res.append(os.path.join(opt['base_dir'], opt['dataset'] + mid_path +opt[key][i]))
    else:
        res = os.path.join(opt['base_dir'], opt['dataset'] + mid_path +opt[key])

    return res

def get_scope(opt):
    scope = []
    #- Fusion type
    scope.append('ADD%d' % (1 if opt['addition'] else 0))
    scope.append('WA%d' % (1 if opt['with_multimodal_attention'] else 0))
    
    #- Traning tricks
    scope.append('EBN%d' % (1 if not opt['no_encoder_bn'] else 0))
    scope.append('SS%d' % (1 if opt['scheduled_sampling'] else 0))
    scope.append('WC%d' % (opt['num_category'] if opt['with_category'] else 0))

    #if opt['scheduled_sampling']:
    #    scope.append('SS%d_%d_%d' % (opt['ss_type'], opt['ss_linear'][0], int(100 * opt['ss_linear'][1])))

    #- Modality
    modality = opt['modality'].lower()
    modality_info = []
    skip_info = opt['skip_info']
    if not len(skip_info):
    	skip_info = [0] * len(modality)

    for i, skip in enumerate(skip_info):
    	if skip: continue
    	current_modality = modality[i]
    	modality_info.append('%s%s' % (current_modality.upper(), opt['auxiliary_for_%s'%current_modality]))

    scope.append('-'.join(modality_info))
    if opt['scope']:
    	scope.append(opt['scope'])

    return '_'.join(scope)

def get_scope2(opt):
    scope = []
    #- Traning tricks
    scope.append('EBN%d' % (1 if not opt['no_encoder_bn'] else 0))
    if not opt['na']:
        scope.append('SS%d' % (1 if opt['scheduled_sampling'] else 0))
    scope.append('NDL%d' % opt['num_hidden_layers_decoder'])
    scope.append('WC%d' % (opt['num_category'] if opt['with_category'] else 0))

    #- Modality
    scope.append(opt['modality'].upper())
    if opt['na']:
        if opt['method'] == 'ms':
            scope.append('%s%d' % (opt['method'], opt['multiscale']))
        elif opt['method'] == 'nv':
            scope.append("%s%02d" % (opt['method'], int(10 * opt['nv_weights'][0])))
        else:
            scope.append(opt['method'])

        if opt['scheduled_sampling']:
            scope.append('SS%d' % int(100 * opt['ss_linear'][1]))

        if opt['enhance_input']:
            scope.append('ei%d'%opt['enhance_input'])

        scope.append('beta%03d_%03d' % (int(100 * opt['beta'][0]), int(100 * opt['beta'][1])))

    

    if opt['dist']:
        scope.append('dist%d'%opt['dist'])

    if opt['shared_embedding']:
        scope.append('se')
    if opt['scope']:
        scope.append(opt['scope'])

    return '_'.join(scope)

def print_information(opt, model, model_name):
    print(model)
    print('| model {}'.format(model_name))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))
    print(sum(p.numel() for n, p in model.named_parameters() if 'encoder' in n.lower()))
    print('use trigger: %d' % opt.get('use_trigger', 0))
    print('trigger level: %g' % opt.get('trigger_level', 0.25))
    print('dataloader random type: %s' % opt.get('random_type', 'segment_random'))
    print('k best model: %d' % opt.get('k_best_model', 10))
    print('teacher prob: %g' % opt.get('teacher_prob', 1.0))
    print('save model limit: %d' % opt.get('save_model_limit', -1))
    print('modality: %s' % opt.get('modality', 'ic'))
    print('equally sampling: %s' % opt.get('equally_sampling', False))
    print('n frames: %d' % opt['n_frames'])
    print('start eval epoch: %d' % opt['start_eval_epoch'])
    print('save_checkpoint_every: %d' % opt['save_checkpoint_every'])
    print('max_len: %d' % opt['max_len'])
    print('scheduled_sampling: {}'.format(opt['scheduled_sampling']))
    print('vocab_size: %d' % opt['vocab_size'])

def main(opt):
    if opt.get('seed', -1) == -1:
        opt['seed'] = random.randint(1, 65534)
    utils.set_seed(opt['seed'])
    print('SEEEEEEEEEEED: %d'%opt['seed'])

    model_name = opt['encoder_type'] + '_' + opt['decoder_type']
    modality = opt['modality'].upper()#(opt['modality'].upper() + 's') if 'c3d' in opt['feats_m_name'][0] else opt['modality'].upper()
    
    if opt['na'] or opt['ar']:
        scope = get_scope2(opt)
    else:
        scope = get_scope(opt)

    opt["checkpoint_path"] = os.path.join(opt["checkpoint_path"], opt['checkpoint_path_name'], opt['dataset'], model_name, scope)

    opt['feats_a'] = get_dir(opt, 'feats_a_name', '/feats/')
    opt['feats_m'] = get_dir(opt, 'feats_m_name', '/feats/', pre=True)
    opt['feats_i'] = get_dir(opt, 'feats_i_name', '/feats/', pre=True)
    opt['feats_s'] = get_dir(opt, 'feats_s_name', '/feats/', pre=True)
    opt['feats_t'] = get_dir(opt, 'feats_t_name', '/feats/')#, pre=True)

    '''
    opt['info_json'] = get_dir(opt, 'info_json_name', post='_%d.json' % opt['word_count_threshold'], prefix=opt['prefix'])
    opt['caption_json'] = get_dir(opt, 'caption_json_name', post='_%d.json' % opt['word_count_threshold'], prefix=opt['prefix'])
    opt['next_info_json'] = get_dir(opt, 'next_info_json_name', post='_%d.json' % opt['word_count_threshold'], prefix=opt['prefix'])
    opt['all_caption_json'] = get_dir(opt, 'all_caption_json_name', post='_%d.json' % opt['word_count_threshold'], prefix=opt['prefix'])
    opt['input_json'] = get_dir(opt, 'input_json_name', post='.json')
    '''
    opt['reference'] = get_dir(opt, 'reference_name', post='.pkl', pre=True)

    if opt.get('knowledge_distillation_with_bert', False):
        opt['bert_embeddings'] = get_dir(opt, 'bert_embeddings_name', '/feats/', pre=True)

    opt['info_corpus'] = get_dir(opt, 'info_corpus_name', post='_%d%s.pkl' % (opt['word_count_threshold'], '_%d' % opt['dist'] if opt['dist'] else ''), prefix=opt['prefix'])

    opt['corpus_pickle'] = get_dir(opt, 'corpus_name', post='.pkl')

    opt['vocab_size'] = len(pickle.load(open(opt['corpus_pickle'] if opt['others'] else opt['info_corpus'], 'rb'))['info']['itow'].keys())
    #opt['tag_size'] = len(json.load(open(opt["info_json"]))['ix_to_tag'].keys())


    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))

    model = get_model(opt)
    device = torch.device('cuda' if not opt['no_cuda'] else 'cpu')
    
    '''517 yb'''
    if opt.get('use_beam_decoder', False):
        assert opt['load_pretrained']
        checkpoint = torch.load(opt['load_pretrained'])['state_dict']
        
        # make sure that current network is the same as the pretrained model
        #namelist = [item for item, _ in model.named_parameters()]
        #print(namelist)
        #for k in checkpoint.keys():
        #    if 'bn' in k: 
        #        continue
        #    print(k)
        #    assert k in namelist
        model.load_state_dict(checkpoint, strict=False)

        # we only train beam decoder
        for name, parameter in model.named_parameters():
            if 'beam' not in name:
                parameter.requires_grad = False

        print_information(opt, model, model_name)
        train_beam_decoder(opt, model, device, first_evaluate_whole_folder=opt['first_evaluate_whole_folder'])
    else:
        if opt['load_pretrained']:
            model.load_state_dict(torch.load(opt['load_pretrained'])['state_dict'])
            
        print_information(opt, model, model_name)
        train_network_all(opt, model, device, first_evaluate_whole_folder=opt['first_evaluate_whole_folder'])



if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    main(opt)

