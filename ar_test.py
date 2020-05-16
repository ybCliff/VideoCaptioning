''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import os,json
from misc.run import get_model, get_loader, run_eval
from misc.logger import CsvLogger

my_mapping = {}
content = [
    [["``", "''", ",", "-LRB-", "-RRB-", ".", ":", "HYPH", "NFP"], "PUNCT"],
    [["$", "SYM"], "SYM"],
    [["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"], "VERB"],
    [["WDT", "WP$", "PRP$", "DT", "PDT"], "DET"],
    [["NN", "NNP", "NNPS", "NNS"], "NOUN"],
    [["WP", "EX", "PRP"], "PRON"],
    [["JJ", "JJR", "JJS", "AFX"], "ADJ"],
    [["ADD", "FW", "GW", "LS", "NIL", "XX"], "X"],
    [["SP", "_SP"], "SPACE"], 
    [["RB", "RBR", "RBS","WRB"], "ADV"], 
    [["IN", "RP"], "ADP"], 
    [["CC"], "CCONJ"],
    [["CD"], "NUM"],
    [["POS", "TO"], "PART"],
    [["UH"], "INTJ"]
]
for item in content:
    ks, v = item
    for k in ks:
        my_mapping[k] = v

def loop_category(option, opt, model, device):
    loop_logger = CsvLogger(filepath='./category_results', filename='ARVC_%s%s.csv' % (option['dataset'], '' if option.get('method', None) != 'ag' else '_ag'), 
                fieldsnames=['novel', 'unique', 'usage', 'ave_length', 'gram4'])
    for i in range(20):
        loader = get_loader(option, mode=opt.em, specific=i)
        vocab = loader.dataset.get_vocab()
        metric = run_eval(option, model, None, loader, vocab, device, print_sent=opt.print_sent, no_score=True, analyze=True)
        loop_logger.write(metric)


def collect_analyze(opt):
    import pickle, nltk
    '''
    import spacy
    from collections import Counter

    data = pickle.load(open(os.path.join(opt.pickle_path), 'rb'))

    all_sents = []
    for key in data.keys():
        for item in data[key]:
            all_sents.append(item['caption'])

    noun_verb_set = set()
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(' '.join(all_sents))
    for token in doc:
        if token.pos_ in ['VERB', 'NOUN'] and str(token) not in ['is', 'are', 'unk']:
            noun_verb_set.add(token.text)

    print('TOP %d:' % opt.topk)
    print('\t--Noun/verb coverage:', len(noun_verb_set))


    all_sents = []
    for key in data.keys():
        all_sents.append(data[key][0]['caption'])

    noun_verb_set = set()
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(' '.join(all_sents))
    for token in doc:
        if token.pos_ in ['VERB', 'NOUN'] and str(token) not in ['is', 'are', 'unk']:
            noun_verb_set.add(token.text)

    print('TOP 1:')
    print('\t--Noun/verb coverage:', len(noun_verb_set))
    
    #pth = os.path.join(opt.collect_path, opt.collect.split('.')[0]+'_last_nv.txt')
    #with open(pth, 'w') as f:
    #    f.write('\n'.join(sorted(list(noun_verb_set))))
    '''
    data = pickle.load(open(os.path.join(opt.pickle_path), 'rb'))
    
    noun_verb_set = set()
    all_word_set = set()

    for key in data.keys():
        for item in data[key]:
            cap = item['caption'].split(' ')
            tag_res = nltk.pos_tag(cap)
            for p, (w, t) in enumerate(tag_res):
                tag = my_mapping[t]
                if tag in ['VERB', 'NOUN'] and w not in ['is', 'are', '<mask>']:
                    noun_verb_set.add(w)
                all_word_set.add(w)
    
    all_unique_nv = len(noun_verb_set)
    all_unique_all = len(all_word_set)

    print('All candidates:')
    print('\t--Noun/verb coverage:', all_unique_nv)
    print('\t--Vocabulary coverage', all_unique_all)

    pth = os.path.join(opt.pickle_path.split('.')[0] + '_all_vocab.txt')
    with open(pth, 'w') as f:
        f.write('\n'.join(sorted(list(all_word_set))))


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model_path', nargs='+', type=str, default=[
                "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_ARFormer/EBN1_SS0_NDL1_WC0_MI/",
                #"/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_MI/",
                "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_MI_seed920/",
                "/home/yangbang/VideoCaptioning/0219save/VATEX/IEL_ARFormer/EBN1_SS1_NDL1_WC0_M/",
                "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_MI_seed1314_ag/",
            ]
        )
    parser.add_argument('-model_name', nargs='+', type=str, default=[
                '0044_240095_254102_253703_251149_247202.pth.tar',
                #'0028_177617_180524_183734_183213_182417.pth.tar',
                "0011_176183_177176_180332_180729_178864.pth.tar",
                '0099_160093_057474.pth.tar',
                "0025_179448_180500_184018_184037_182508.pth.tar",
            ]
        )
    parser.add_argument('-i', '--index', default=0, type=int)

    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-beam_alpha', type=float, default=1.0)
    parser.add_argument('-batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('-topk', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-em', type=str, default='test')
    parser.add_argument('-print_sent', action='store_true')
    parser.add_argument('-json_path', type=str, default='')
    parser.add_argument('-json_name', type=str, default='')
    parser.add_argument('-ns', default=False, action='store_true')
    parser.add_argument('-sv', default=False, action='store_true')
    parser.add_argument('-analyze', default=False, action='store_true')
    parser.add_argument('-write_time', default=False, action='store_true')
    parser.add_argument('-mid_path', default='best', type=str)
    parser.add_argument('-specific', default=-1, type=int)
    parser.add_argument('-category', default=False, action='store_true')
    parser.add_argument('-sp', '--saved_with_pickle', default=False, action='store_true')
    parser.add_argument('-pp', '--pickle_path', default='./AR_topk_collect_results')
    parser.add_argument('-ca', '--collect_analyze', default=False, action='store_true')

    parser.add_argument('-cv', '--cross_validation', type=int, default=2)
    

    opt = parser.parse_args()
    opt.model_path = opt.model_path[opt.index]
    opt.model_name = opt.model_name[opt.index]
    if opt.cross_validation == 1:
        source_dataset = 'MSRVTT'
        src_pre = 'msrvtt'
        src_wct = '2'
        target_dataset = 'Youtube2Text'
        tgt_pre = 'msvd'
        tgt_wct = '0'
    else:
        source_dataset = 'Youtube2Text'
        src_pre = 'msvd'
        src_wct = '0'
        target_dataset = 'MSRVTT'
        tgt_pre = 'msrvtt'
        tgt_wct = '2'

    opt_pth = os.path.join(opt.model_path, 'opt_info.json')
    option = json.load(open(opt_pth, 'r'))
    option.update(vars(opt))
    if opt.saved_with_pickle:
        if not os.path.exists(opt.pickle_path):
            os.makedirs(opt.pickle_path)
        string = ''
        if 'Youtube2Text' in opt.model_path:
            dataset_name = 'msvd'
        elif 'MSRVTT' in opt.model_path:
            dataset_name = 'msrvtt'
        if option.get('method', None) == 'ag':
            string = '_ag'
        opt.pickle_path = os.path.join(opt.pickle_path, '%s_%d%s.pkl' % (dataset_name, opt.topk, string))



    if opt.collect_analyze:
        collect_analyze(opt)
    else:
        device = torch.device('cuda' if not opt.no_cuda else 'cpu')
        if opt.analyze:
            opt.batch_size = 1
            option['batch_size'] = 1

        
        
        #print(option)
        
        checkpoint = torch.load(os.path.join(opt.model_path, opt.mid_path, opt.model_name))
        #option = checkpoint['settings']
        #option.update(vars(opt))
        model = get_model(option)
        
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)

        if opt.category:
            loop_category(option, opt, model, device)
        else:
            loader = get_loader(option, mode=opt.em, print_info=True, specific=opt.specific)
            vocab = loader.dataset.get_vocab()

            #print(model)
            calculate_novel = True
            if opt.cross_validation != 2:
                option['dataset'] = target_dataset
                option['feats_a'] = []
                option['feats_m'] = [item.replace(source_dataset, target_dataset) for item in option['feats_m']]
                option['feats_m'] = [item.replace(src_pre, tgt_pre) for item in option['feats_m']]
                option['feats_i'] = [item.replace(source_dataset, target_dataset) for item in option['feats_i']]
                option['feats_i'] = [item.replace(src_pre, tgt_pre) for item in option['feats_i']]
                
                option['reference'] = option['reference'].replace(source_dataset, target_dataset)
                option['reference'] = option['reference'].replace(src_pre, tgt_pre)

                option['info_corpus'] = option['info_corpus'].replace(source_dataset, target_dataset)
                option['info_corpus'] = option['info_corpus'].replace(src_wct, tgt_wct)
                option['info_corpus'] = option['info_corpus'].replace('Youtube0Text', 'Youtube2Text')

                loader = get_loader(option, mode=opt.em, print_info=True, specific=opt.specific)
                calculate_novel = False

            print(len(vocab))
            
            metric = run_eval(option, model, None, loader, vocab, device, json_path=opt.json_path, json_name=opt.json_name, print_sent=opt.print_sent, 
                no_score=opt.ns, save_videodatainfo=opt.sv, analyze=opt.analyze, saved_with_pickle=opt.saved_with_pickle, pickle_path=opt.pickle_path, write_time=opt.write_time, calculate_novel=calculate_novel)
            
            print(metric)
        

if __name__ == "__main__":
    main()

'''
CUDA_VISIBLE_DEVICES=3 python ar_test.py -i 1 -em test -beam_size 5 -topk 5 -sv -ns
CUDA_VISIBLE_DEVICES=3 python ar_test.py -i 1 -em test -beam_size 5 -topk 5 -sv -ns -sp
'''