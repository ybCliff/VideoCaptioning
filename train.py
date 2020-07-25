import json
import os
from opts import parse_opt
import torch
import misc.utils as utils
from models import get_model
from misc.run import train_network_all, train_beam_decoder
import random
import pickle


def get_dir(opt, key, mid_path=''):
    if not opt.get(key, ''):
        return ''
    res = []
    if isinstance(opt[key], list):
        if not opt[key][0]:
            return ''
        for i in range(len(opt[key])):
            res.append(os.path.join(opt['base_dir'], opt['dataset'], mid_path, opt[key][i]))
    else:
        res = os.path.join(opt['base_dir'], opt['dataset'], mid_path, opt[key])
    return res


def where_to_save_model(opt):
    model_name = opt['encoder_type'] + '_' + opt['decoder_type']

    scope = [opt['modality'].upper(), 'BN%d' % (1 if not opt['no_bn'] else 0),
             'SS%d' % (1 if opt['scheduled_sampling'] else 0),
             'WC%d' % (opt['num_category'] if opt['with_category'] else 0)]
    scope += opt['scope'] if opt['scope'] else []
    scope = '_'.join(scope)

    return os.path.join(
        opt["checkpoint_path"],
        opt['dataset'],
        model_name,
        scope
    )


def print_information(model):
    print(model)
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))
    print(sum(p.numel() for n, p in model.named_parameters() if 'encoder' in n.lower()))


def main(opt):
    utils.set_seed(opt['seed'] if opt['seed'] != -1 else random.randint(1, 65534))

    opt["checkpoint_path"] = where_to_save_model(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])

    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))

    key_list = ['feats_a', 'feats_m', 'feats_i', 'feats_s', 'reference', 'info_corpus']
    for key in key_list:
        opt[key] = get_dir(opt, key + '_name', mid_path='feats' if 'feats' in key else '')

    opt['vocab_size'] = len(pickle.load(open(opt['info_corpus'], 'rb'))['info']['itow'].keys())
    model = get_model(opt)
    print_information(model)
    device = torch.device('cuda' if not opt['no_cuda'] else 'cpu')
    model = model.to(device)

    train_network_all(opt, model, device, first_evaluate_whole_folder=opt['first_evaluate_whole_folder'])


if __name__ == '__main__':
    opt = parse_opt()
    opt = vars(opt)
    main(opt)
