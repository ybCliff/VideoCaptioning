''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import os,json
from misc.run import get_model, get_loader, get_forword_results, run_eval, cal_score, get_dict_mapping
from misc.logger import CsvLogger
from misc.utils import set_seed
import matplotlib.pyplot as plt
import pickle
import nltk
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


'''
def loop(option, model, loader, device, teacher_model, dict_mapping, filepath, filename):
    loop_logger = CsvLogger(filepath=filepath, filename=filename+'.csv', 
                fieldsnames=['lbs', 'i', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'Sum', 'novel', 'unique', 'usage', 'ave_length'])
    best_res = {'Sum': 0}
    for lbs in range(1, 16):
        for i in range(1, 20):
            option['length_beam_size'] = lbs
            option['iterations'] = i
            metric = run_eval(option, model, None, loader, loader.dataset.get_vocab(), device, teacher_model=teacher_model, dict_mapping=dict_mapping, analyze=True)
            metric.pop('loss')
            metric['i'] = i
            metric['lbs'] = lbs
            loop_logger.write(metric)

            if metric['Sum'] > best_res['Sum']:
                best_res['Sum'] = metric['Sum']
                best_res['lbs'] = lbs
                best_res['i'] = i

    return best_res



def loop_iterations(option, opt, model, device, teacher_model, dict_mapping):
    keys1, values1 = ['no_duplicate', 'no_duplicate'], [True, False]
    keys2, values2 = ['with_teacher', 'no_teacher'], [False, True]
    keys3, values3 = ['masking_decision'] * 2, [False, True]

    vali_loader = get_loader(option, mode='validate')
    test_loader = get_loader(option, mode='test')

    filepath = './nar_results/' + opt.loop
    loop_logger = CsvLogger(filepath=filepath, filename='test_scores.csv', 
                fieldsnames=['scope', 'Sum', 'lbs', 'i', 'novel', 'unique', 'usage', 'ave_length'])

    for k1,v1 in zip(keys1, values1):
        for k2,v2 in zip(keys2, values2):
            option[k1] = v1
            if v2:
                teacher = teacher_model
                for k3, v3 in zip(keys3, values3):
                    option[k3] = v3
                    filename = 'nd%d_wt%d_md%d' % (int(v1), int(v2), int(v3))
                    best_res = loop(
                        option = option,
                        model = model,
                        loader = vali_loader,
                        device = device,
                        teacher_model = teacher,
                        dict_mapping = dict_mapping,
                        filepath = filepath,
                        filename = filename
                        )
                    best_res['scope'] = filename
                    option['length_beam_size'] = best_res['lbs']
                    option['iterations'] = best_res['i']
                    res = run_eval(option, model, None, test_loader, test_loader.dataset.get_vocab(), device, 
                        json_path=filepath, json_name='%s%s.json'%('Sum%05d_lbs%02d_i%02d_' % (int(10000*best_res['Sum']), best_res['lbs'], best_res['i']), filename), 
                            analyze=True, no_score=True)
                    best_res.update(res)
                    loop_logger.write(best_res)

            else:
                teacher = None
                filename = 'nd%d_wt0' % (int(v1))
                best_res = loop(
                        option = option,
                        model = model,
                        loader = vali_loader,
                        device = device,
                        teacher_model = teacher,
                        dict_mapping = dict_mapping,
                        filepath = filepath,
                        filename = filename
                        )
                best_res['scope'] = filename
                option['length_beam_size'] = best_res['lbs']
                option['iterations'] = best_res['i']
                res = run_eval(option, model, None, test_loader, test_loader.dataset.get_vocab(), device, 
                        json_path=filepath, json_name='%s%s.json'%('Sum%05d_lbs%02d_i%02d_' % (int(10000*best_res['Sum']), best_res['lbs'], best_res['i']), filename), 
                            analyze=True, no_score=True)
                best_res.update(res)
                loop_logger.write(best_res)

'''   


def loop_iterations(option, opt, model, loader, vocab, device, teacher_model, dict_mapping):
    loop_logger = CsvLogger(filepath='./loop_results', filename=opt.loop + '.csv', 
                fieldsnames=['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'Sum', 'iterations', 'lbs', 'novel', 'unique', 'usage', 'ave_length'])
    for i in range(1, 11):
        option['iterations'] = i
        metric = run_eval(option, model, None, loader, vocab, device, json_path=opt.json_path, json_name=opt.json_name, 
                print_sent=opt.print_sent, teacher_model=teacher_model, length_crit=torch.nn.SmoothL1Loss(),
                dict_mapping=dict_mapping, analyze=True)

        metric['iterations'] = option['iterations']
        metric['lbs'] = option['length_beam_size']
        metric.pop('loss')
        loop_logger.write(metric)

def loop_length_beam(option, opt, model, loader, vocab, device, teacher_model, dict_mapping):
    b4 = []
    m = []
    r = []
    c = []
    ave_len = []  
    for lbs in range(1, 11):
        option['length_beam_size'] = lbs
        metric = run_eval(option, model, None, loader, vocab, device, json_path=opt.json_path, json_name=opt.json_name, 
                print_sent=opt.print_sent, teacher_model=teacher_model, length_crit=torch.nn.SmoothL1Loss(),
                dict_mapping=dict_mapping, analyze=True)
        b4.append(metric["Bleu_4"])
        m.append(metric["METEOR"])
        r.append(metric["ROUGE_L"])
        c.append(metric["CIDEr"])
        ave_len.append(metric['ave_length'])

    print(b4)
    print(m)
    print(r)
    print(c)
    print(ave_len)

def loop_category(option, opt, model, device, teacher_model, dict_mapping):
    loop_logger = CsvLogger(filepath='./category_results', filename='NAVC_%s_%s%s.csv' % (option['method'], 'AE' if opt.nv_scale == 100 else '', opt.paradigm), 
                fieldsnames=['novel', 'unique', 'usage', 'ave_length', 'gram4'])
    for i in range(20):
        loader = get_loader(option, mode=opt.em, specific=i)
        vocab = loader.dataset.get_vocab()
        metric = run_eval(option, model, None, loader, vocab, device, print_sent=opt.print_sent, no_score=True, analyze=True, teacher_model=teacher_model, dict_mapping=dict_mapping)
        loop_logger.write(metric)



def load(checkpoint_path, checkpoint_name, device, mid_path='', opt_name='opt_info.json', from_checkpoint=False):
    checkpoint = torch.load(os.path.join(checkpoint_path, mid_path, checkpoint_name))
    if from_checkpoint:
        opt = checkpoint['settings']
    else:
        opt = json.load(open(os.path.join(checkpoint_path, opt_name)))
    model = get_model(opt)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model, opt

def plot(option, opt, model, loader, vocab, device, teacher_model, dict_mapping):
    colors=['skyblue', 'dodgerblue', 'dodgerblue', 'dodgerblue', 'dodgerblue']
    for i, iteration in enumerate([1, 2, 3, 4, 5]):
        option['iterations'] = iteration
        metric, x, y = run_eval(option, model, None, loader, vocab, device, json_path=opt.json_path, json_name=opt.json_name, 
                print_sent=opt.print_sent, teacher_model=teacher_model, length_crit=torch.nn.SmoothL1Loss(),
                dict_mapping=dict_mapping, length_bias=opt.lb, analyze=True, plot=True, no_score=True, top_n=30)

        ax = plt.subplot(1, 5, i+1)
        ax.barh(x, y, color=colors[i])
        ax.set_title('iteration=%d'%iteration)
            #plt.tick_params(labelsize=13)
    plt.subplots_adjust(left=0.12, bottom=None, right=0.98, top=None, wspace=0.4, hspace=None)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.savefig('./iteration.png')
    plt.show()
    

def main(opt):
    '''Main Function'''
    if opt.collect:
        if not os.path.exists(opt.collect_path):
            os.makedirs(opt.collect_path)
    
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')

    model, option = load(opt.model_path, opt.model_name, device, mid_path='best')
    option.update(vars(opt))
    set_seed(option['seed'])

    if not opt.nt:
        #teacher_path = os.path.join(option["checkpoint_path"].replace('NARFormer', 'ARFormer') + '_SS1_0_70')
        #teacher_name = 'teacher.pth.tar'
        #teacher_model, teacher_option = load(teacher_path, teacher_name, device, mid_path='', from_checkpoint=True)
        
        
        checkpoint = torch.load(opt.teacher_path)
        teacher_option = checkpoint['settings']
        teacher_model = get_model(teacher_option)
        teacher_model.load_state_dict(checkpoint['state_dict'])
        teacher_model.to(device)
    
        assert teacher_option['vocab_size'] == option['vocab_size']
        
        #dict_mapping = get_dict_mapping(option, teacher_option)
        dict_mapping = {}
    else:
        teacher_model = None
        dict_mapping = {}


    '''
    model = get_model(option)
    pth = os.path.join(opt.model_path, 'tmp_models')
    vali_loader = get_loader(option, mode='validate')
    test_loader = get_loader(option, mode='test')
    vocab = vali_loader.dataset.get_vocab()
    logger = CsvLogger(
        filepath=pth, 
        filename='evaluate.csv', 
        fieldsnames=['epoch', 'split', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'Sum', 'lbs', 'i', 'ba']
        )
    for file in os.listdir(pth):
        if '.pth.tar' not in file:
            continue
        epoch = file.split('_')[1]
        checkpoint = torch.load(os.path.join(pth, file))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)

        best = 0
        best_info = ()
        for lbs in range(1, 11):
            for i in [1, 3, 5, 10]:
                option['length_beam_size'] = lbs
                option['iterations'] = i
                metric = run_eval(option, model, None, test_loader, vocab, device, json_path=opt.json_path, json_name=opt.json_name, print_sent=opt.print_sent)
                metric.pop('loss')
                metric['lbs'] = lbs
                metric['ba'] = opt.beam_alpha
                metric['i'] = i
                metric['split'] = 'test'
                metric['epoch'] = epoch
                logger.write(metric)
                if metric['Sum'] > best:
                    best = metric['Sum']
                    best_info = (lbs, i)
                print(lbs, i, metric['Sum'], best)


    '''    

    


    '''
    # rec length predicted results
    rec = {}
    for data in tqdm(loader, ncols=150, leave=False):
        with torch.no_grad():
            results = get_forword_results(option, model, data, device=device, only_data=False)
            for i in range(results['pred_length'].size(0)):
                res = results['pred_length'][i].topk(5)[1].tolist()
                for item in res:
                    rec[item] = rec.get(item, 0) + 1
    for i in range(50):
        if i in rec.keys():
            print(i, rec[i])
    '''
    if opt.plot:
        plot(option, opt, model, loader, vocab, device, teacher_model, dict_mapping)
    elif opt.loop:
        loader = get_loader(option, mode=opt.em, print_info=True)
        vocab = loader.dataset.get_vocab()
        #loop_iterations(option, opt, model, loader, vocab, device, teacher_model, dict_mapping)
        loop_length_beam(option, opt, model, loader, vocab, device, teacher_model, dict_mapping)
        #loop_iterations(option, opt, model, device, teacher_model, dict_mapping)
    elif opt.category:
        loop_category(option, opt, model, device, teacher_model, dict_mapping)
    else:
        loader = get_loader(option, mode=opt.em, print_info=True, specific=opt.specific)
        vocab = loader.dataset.get_vocab()
        filename = '%s_%s_%s_i%db%da%03d%s.pkl' % (option['dataset'], option['method'], ('%s' % ('AE' if opt.nv_scale == 100 else '')) + opt.paradigm, opt.iterations, opt.length_beam_size, int(100*opt.beam_alpha), '_all' if opt.em =='all' else '')
        metric = run_eval(option, model, None, loader, vocab, device, json_path=opt.json_path, json_name=opt.json_name, 
            print_sent=opt.print_sent, teacher_model=teacher_model, length_crit=torch.nn.SmoothL1Loss(),
            dict_mapping=dict_mapping, analyze=opt.analyze,
            collect_best_candidate_iterative_results=True if opt.collect else False,
            collect_path=os.path.join(opt.collect_path, filename),
            no_score=opt.ns,
            write_time=opt.write_time
            )
        #collect_path=os.path.join(opt.collect_path, opt.collect),
        print(metric)
    
def test(opt):
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')
    model, option = load(opt.model_path, opt.model_name, device, mid_path='tmp_models')
    cal_score(option)

def collect_analyze(opt):
    import spacy
    from collections import Counter
    data = pickle.load(open(os.path.join(opt.collect_path, opt.collect_analyze), 'rb'))
    sents, scores = data
    
    keys = ['i', 'unique', 'VERB', 'NOUN', 'ADP', 'DET', 'SYM', 'PUNCT', 'X', 'ADJ', 'CCONJ', 'NUM', 'PRON', 'PROPN', 'PART', 'ADV', 'SPACE', 'INTJ']
    '''
    logger = CsvLogger(
        filepath=opt.collect_path, 
        filename=opt.collect.split('.')[0] + '_pos_results.csv', 
        fieldsnames=keys
        )
    '''
    keylist = sents.keys()
    
    noun_verb_set = set()
    all_word_set = set()
    for key in keylist:
        cap = sents[key][-1].split(' ')
        tag_res = nltk.pos_tag(cap)
        for p, (w, t) in enumerate(tag_res):
            tag = my_mapping[t]
            if tag in ['VERB', 'NOUN'] and w not in ['is', 'are']:
                noun_verb_set.add(w)
            all_word_set.add(w)

    
    last_unique_nv = len(noun_verb_set)
    last_unique_all = len(all_word_set)

    print('VERB NOUN in the last iteration:')
    print('\t--Noun/verb coverage:', last_unique_nv)
    print('\t--Vocabulary coverage', last_unique_all)

    pth = os.path.join(opt.collect_path, opt.collect_analyze.split('.')[0]+'_last_vocab.txt')
    with open(pth, 'w') as f:
        f.write('\n'.join(sorted(list(all_word_set))))

    noun_verb_set = set()
    all_word_set = set()
    for key in keylist:
        for j in range(len(sents[key])):
            cap = sents[key][j].split(' ')
            tag_res = nltk.pos_tag(cap)
            for p, (w, t) in enumerate(tag_res):
                tag = my_mapping[t]
                if tag in ['VERB', 'NOUN'] and w not in ['is', 'are', '<mask>']:
                    noun_verb_set.add(w)
                all_word_set.add(w)

    
    all_unique_nv = len(noun_verb_set)
    all_unique_all = len(all_word_set)

    print('VERB NOUN in all iterations:')
    print('\t--Noun/verb coverage:', all_unique_nv)
    print('\t--Vocabulary coverage', all_unique_all)

    with open(os.path.join(opt.collect_path, opt.collect_analyze.split('.')[0]+'_all_vocab.txt'), 'w') as f:
        f.write('\n'.join(sorted(list(all_word_set))))

    with open(os.path.join(opt.collect_path, opt.collect_analyze.split('.')[0]+'_all_vocab_nv.txt'), 'w') as f:
        f.write('\n'.join(sorted(list(noun_verb_set))))

    '''
    #iterative pos results
    for i in range(iterations):
        doc = ' '.join([item[1] for item in sents[i]])
        doc = nlp(doc)
        pos = [item.pos_ for item in doc]

        default_res = {k: 0 for k in keys}
        default_res['i'] = i
        count_res = dict(Counter(pos))
        default_res.update(count_res)

        default_res['unique'] = unique
        default_res['confidence'] = confidence
        default_res['backtracking steps'] = steps

        logger.write(default_res)
        print(default_res)
    '''
  

def compare_analyze(opt):
    '''
    sents1, scores1 = pickle.load(open(os.path.join(opt.collect_path, opt.compare[0]), 'rb'))
    sents2, scores2 = pickle.load(open(os.path.join(opt.collect_path, opt.compare[1]), 'rb'))
    
    assert len(sents2) == len(sents1) + 1
    assert len(sents2[0]) == len(sents1[0])
    
    num_sample = len(sents1[0])
    i1 = len(sents1)
    i2 = len(sents2)

    for i in range(num_sample):
        if sents1[0][i][1] != sents2[1][i][1] and sents1[-1][i][1] != sents2[-1][i][1]:
            vid = sents1[0][i][0]
            assert vid == sents2[1][i][0]
            for j in range(i1):
                print('MP %10s:'%vid, sents1[j][i][1])
            for j in range(i2):
                print('NV %10s:'%vid, sents2[j][i][1])
            print('-------------------------')
    '''
    sents, scores = pickle.load(open(os.path.join(opt.collect_path, opt.compare[0]), 'rb'))
    key = 'video8507'
    for sent, score in zip(sents[key], scores[key]):
        print(sent, score)




def analyze_six_algorithms(opt):
    sents = []
    scores = []
    names = []
    dataset = 'MSRVTT'
    method = 'nv'
    algorithm = ['mp', 'ef', 'l2r']
    AE_or_not = ['AE', '']
    for i in AE_or_not:
        for j in algorithm:
            filename = '%s_%s_%s_i%db%da%03d.pkl' % (dataset, method, i + j, 1 if i == 'AE' and j in ['ef', 'l2r'] else opt.iterations, opt.length_beam_size, int(100*opt.beam_alpha))
            data = pickle.load(open(os.path.join(opt.collect_path, filename), 'rb'))
            sents.append(data[0])
            scores.append(data[1])
            names.append(i+j)

    keylist = sents[0].keys()
    for k in keylist:
        if k != 'video8507':
            continue
        length = len(sents[0][k][0].split(' '))
        jud = True
        for j in range(1, len(sents)):
            if len(sents[j][k][0].split(' ')) != length:
                jud=False
                break
        #print(k, jud)
        if jud and sents[3][k][0] != sents[3][k][1]: #and k == 'video1877':# and k == 'video1922' : #'video1779'
            jud2 = True
            s1 = sents[0][k][0].split(' ')
            s2 = sents[3][k][0].split(' ')
            for w1, w2 in zip(s1, s2):
                if w1 == '<mask>':
                    continue
                if w1 != w2:
                    jud2 = False
            if not jud2:
                for j in range(len(sents)):
                    for sent in sents[j][k]:
                        print(k, '%6s' % names[j], sent)
                print('----------------')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model_path', nargs='+', type=str, default=[
                #"/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_NARFormer/EBN1_NDL1_WC0_MI_mp_ei2/",
                #"/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_NARFormer/EBN1_NDL1_WC0_MI_nv_ei2_w10/",
                #"/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_NARFormer/EBN1_NDL1_WC0_MI_nv08_ei2/",
                "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_NARFormer/EBN1_NDL1_WC0_MI_nv08_ei2_mpri5/",

                #"/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC20_MI_nv_ei2_w08/",
                #"/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC20_MI_nv08_ei2/",
                "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC20_MI_nv08_ei2_mp0.35_0.9/",
                #"/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC0_MI_nv08_ei2/",
                #"/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC20_MI_nv08_beta035_090/",
                #"/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC0_MI_nv08_ei2_beta035_090/",
                #"/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC20_MI_nv08_ei2_beta035_090_noun/",
                #"/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC20_MI_nv08_ei2_beta035_090_verb/",
                #"/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC20_MI_nv08_ei2_beta035_090_reverse/",

                "/home/yangbang/VideoCaptioning/0219save/VATEX/IEL_NARFormer/EBN1_NDL1_WC0_M_nv_ei2_w08/",
                "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_NARFormer/EBN1_NDL1_WC0_I_nv_ei2/",
                "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_NARFormer/EBN1_NDL1_WC0_M_nv_ei2/",

                "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC20_I_nv_ei2_w08/",
                "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC20_M_nv_ei2_w08/",

                "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_NARFormer/EBN1_NDL1_WC0_MI_nv08_ei2_eco2/",
                "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC20_MI_nv08_beta035_090_AudioEmb/",

        "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_NARFormer/EBN1_NDL1_WC0_MI_mp_ei2/",
        "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC20_MI_mp_ei2_beta035_090/",
            ]
        )
    parser.add_argument('-model_name', nargs='+', type=str, default=[
                #'0040_098890_235779_241391_242726_248156_244386.pth.tar',
                #'0040_098349_240876_253836_252135_253562_250689.pth.tar',
                #'0040_098151_248121_249221_251796_250427_250961.pth.tar',
                "0043_099076_245587_251188_252255_248551_247855.pth.tar",

                #'0041_099560_141436_166370_171335_175096_175373.pth.tar',
                #"0049_099603_151397_167075_171060_178066_174592.pth.tar",
                "0027_099736_153026_167039_170898_177000_175692.pth.tar",
                #"0031_099399_154607_165454_171273_174564_171729.pth.tar",
                #"0048_100000_157487_168288_172855_177611_172560.pth.tar",
                #"0046_099271_156077_166886_172315_173689_172102.pth.tar",
                #"0047_098275_148785_163007_171671_177010_173522.pth.tar",
                #"0022_099576_135114_156339_164823_172876_168686.pth.tar",
                #"0039_098986_143613_162035_168945_172163_170912.pth.tar",
                
                '0067_149417_051040.pth.tar',
                '0021_098668_219703_225013_227193_228162_222878.pth.tar',
                '0032_097046_227743_233650_234877_234121_226178.pth.tar',

                '0040_099313_139109_162610_166375_166314_167868.pth.tar',
                '0048_099871_136524_163501_168170_168757_167925.pth.tar',

                "0049_099962_227137_229961_229627_230691_232494.pth.tar",
                "0046_099501_155578_172153_176020_179829_175984.pth.tar",

        '0040_098890_235779_241391_242726_248156_244386.pth.tar',
        '0028_099964_127192_165202_170834_171675_170648.pth.tar',
            ]
        )
    parser.add_argument('-teacher_path', nargs='+', type=str, default=[
                #"/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_ARFormer/EBN1_SS0_NDL1_WC0_MI/best/0027_239738_254857_255009_253614_249835.pth.tar",
                "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_ARFormer/EBN1_SS0_NDL1_WC0_MI/best/0044_240095_254102_253703_251149_247202.pth.tar",
                #'/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_MI/best/0044_178764_179682_183792_184948_184293.pth.tar',
                
                #"/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_MI/best/0028_177617_180524_183734_183213_182417.pth.tar",
                "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_MI_seed920/best/0011_176183_177176_180332_180729_178864.pth.tar",

                #"/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS1_NDL1_WC0_MI/best/0039_175275_177498_180276_180783_179707.pth.tar",
                #"/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_MI/best/0041_177384_179698_182822_183440_183007.pth.tar",
                #"/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS1_NDL1_WC20_MI/best/0024_178148_179210_183009_183345_183243.pth.tar",
                "/home/yangbang/VideoCaptioning/0219save/VATEX/IEL_ARFormer/EBN1_SS1_NDL1_WC0_M/best/0099_160093_057474.pth.tar",
                "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_ARFormer/EBN1_SS0_NDL1_WC0_I/best/0016_221322_228580_230261_226309_223136.pth.tar",
                "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_ARFormer/EBN1_SS0_NDL1_WC0_M/best/0023_226212_232464_234939_234077_232347.pth.tar",

                "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_I/best/0036_168911_172305_174303_175478_175142.pth.tar",
                "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_M/best/0039_171764_175967_179945_179093_177663.pth.tar",

                "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_ARFormer/EBN1_SS0_NDL1_WC0_MI_eco2/best/0012_220987_231395_232369_229921_227882.pth.tar",
                "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_MI/best/0028_177617_180524_183734_183213_182417.pth.tar",
        
        "/home/yangbang/VideoCaptioning/0219save/Youtube2Text/IEL_ARFormer/EBN1_SS0_NDL1_WC0_MI/best/0044_240095_254102_253703_251149_247202.pth.tar",
        "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_MI_seed920/best/0011_176183_177176_180332_180729_178864.pth.tar",
            ]
        )
    #"/home/yangbang/VideoCaptioning/0219save/VATEX/IEL_NARFormer/EBN1_NDL1_WC0_M_nv_ei2_w08_PAD/best/0057_149877_051223.pth.tar"

    parser.add_argument('--index', default=0, type=int)

    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-beam_alpha', type=float, default=1.0)
    parser.add_argument('-batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-em', type=str, default='test')
    parser.add_argument('-print_sent', action='store_true')
    parser.add_argument('-func', type=int, default=1)
    parser.add_argument('-json_path', type=str, default='')
    parser.add_argument('-json_name', type=str, default='')
    parser.add_argument('-js')
    parser.add_argument('-i', '--iterations', type=int, default=5)
    parser.add_argument('-lbs', '--length_beam_size', type=int, default=5)
    parser.add_argument('-nt', default=False, action='store_true')
    parser.add_argument('-nd', '--no_duplicate', default=False, action='store_true')
    parser.add_argument('-ns', default=False, action='store_true')
    #parser.add_argument('-teacher_path', type=str, default="/home/yangbang/VideoCaptioning/1007save/MSRVTT/IPE_ARFormer/AMIs_Seed0_EBN_WC20_SS1_100_70_FromScratchNoWatch_pjDrop/")
    #parser.add_argument('-teacher_name', type=str, default='0_0455_186967_192003_182832_192057_191210.pth.tar')
    #"/home/yangbang/VideoCaptioning/1022save/Youtube2Text/IPE_NARFormer/M_Seed0_EBN_t20_ei2_b64/best/.pth.tar"
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-analyze', default=False, action='store_true')
    parser.add_argument('-print_latency', default=False, action='store_true')
    parser.add_argument('-plot', default=False, action='store_true')
    parser.add_argument('-one', '--only_return_one_item', default=False, action='store_true')
    parser.add_argument('-shift', default=0, type=int)
    parser.add_argument('-loop', type=str, default='')

    parser.add_argument('-md', '--masking_decision', default=False, action='store_true')
    parser.add_argument('-ncd', '--no_candidate_decision', default=False, action='store_true')

    parser.add_argument('-collect_path', type=str, default='./iterative_collect_results')
    parser.add_argument('-collect', default=False, action='store_true')
    parser.add_argument('-ca', '--collect_analyze', default='', type=str)
    parser.add_argument('-s', '--nv_scale', type=float, default=0.0)
    parser.add_argument('-fi', '--fixed_iterations', type=int ,default=-1)

    parser.add_argument('-nobc', '--not_only_best_candidate', default=False, action='store_true')

    parser.add_argument('-cl', '--collect_last', default=False, action='store_true')

    parser.add_argument('-paradigm', type=str, default='mp')
    parser.add_argument('-compare', nargs='+', type=str, default=[])
    parser.add_argument('-six', default=False, action='store_true')
    parser.add_argument('-write_time', default=False, action='store_true')
    parser.add_argument('-q', type=int, default=1)

    parser.add_argument('-category', default=False, action='store_true')
    parser.add_argument('-specific', default=-1, type=int)

    parser.add_argument('-lgc', '--load_generated_captions', default=False, action='store_true')
    parser.add_argument('--generated_captions', type=str, default="/home/yangbang/VideoCaptioning/ARVC/AR_topk_collect_results/msrvtt_1.pkl")#"/home/yangbang/VideoCaptioning/ARVC/AR_topk_collect_results/msvd_1.pkl")

    opt = parser.parse_args()


    if opt.not_only_best_candidate:
        if opt.nt:
            opt.collect_path = os.path.join(opt.collect_path, 'nt')
        else:
            opt.collect_path = os.path.join(opt.collect_path, 'all')
    elif opt.collect_last:
        opt.collect_path = os.path.join(opt.collect_path, 'cl')

    opt.model_path = opt.model_path[opt.index]
    opt.model_name = opt.model_name[opt.index]
    opt.teacher_path = opt.teacher_path[opt.index]

    if opt.print_latency:
        opt.batch_size = 1
    if opt.collect_analyze:
        collect_analyze(opt)
    elif len(opt.compare):
        compare_analyze(opt)
    elif opt.six:
        analyze_six_algorithms(opt)
    else:
        main(opt)
    #test(opt)
    #test2()

'''
CUDA_VISIBLE_DEVICES=0 python test_nar.py --index 1 -em test -nd -paradigm mp -i 5 -beam_alpha 1.15 -s 100 -collect
CUDA_VISIBLE_DEVICES=3 python test_nar.py --index 0 -em test -nd -paradigm mp -i 5 -s 100 -collect -nobc -ca Youtube2Text_nv_AEmp_i5b5a100.pkl
CUDA_VISIBLE_DEVICES=3 python test_nar.py --index 0 -em test -nd -paradigm mp -i 5 -collect -nobc -ca Youtube2Text_mp_mp_i5b5a100.pkl
'''
