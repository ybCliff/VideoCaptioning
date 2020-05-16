import re
import json
import argparse
import pandas
from tqdm import tqdm
import os 
import string
from nltk.tokenize import word_tokenize
import models.Constants as Constants
import wget
import nltk
import pickle
from collections import defaultdict
import spacy

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

def check(sent):
    jud = False
    for char in string.punctuation:
        if char in sent:
            jud = True
            sent.remove(char)
    return jud

def build_vocab(trainvids, count_thr):
    # count up the number of words
    counts = {}
    for vid, caps in trainvids.items():
        for caption in caps:
            cap = caption.split(' ')
            for w in cap:
                counts[w] = counts.get(w, 0) + 1
    # cw = sorted([(count, w) for w, count in counts.items()], reverse=True)

    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]

    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)


    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))

    return vocab

def mapping_tag(tag, w):
    return my_mapping[tag]
    '''
    if 'JJ' in tag: #adjective 
        return 'JJ'
    elif 'NN' in tag: #noun
        return 'NN'
    elif 'RB' in tag: #adverb
        return 'RB'
        #elif tag == 'VB' or tag == 'VBD' or tag == 'VBZ' or tag == 'VBP':
    elif 'VB' in tag:
        if w in ['is', 'are']:
            return tag
        return 'VB'
    else:
        return tag
    '''

def read_vocab_file(vocab_file):
    label_num = 0
    vocab = []
    with open(vocab_file, 'r') as f:
        for line in f:
            line = line.strip()
            vocab.append(line)
            label_num += 1
    return vocab, label_num

def get_tag_info(video_caption, my_data=None):
    if my_data is not None:
        f400, label_num = my_data, len(my_data)
    else:
        f400, label_num = read_vocab_file("/home/yangbang/VideoCaptioning/most_frequent_400.txt")
        assert label_num == 400
    tag_info = {}
    for vid in video_caption.keys():
        caps = video_caption[vid]
        tag_info[vid] = [0] * label_num
        for caption in caps:
            cap = caption.split(' ')
            for w in cap:
                if w in f400:
                    pos = f400.index(w)
                    tag_info[vid][pos] = 1
        #print(tag_info[vid])
    return tag_info

def get_next_info(captions, split_info):
    word_next = {}
    for vid, caps in captions.items():
        if int(vid[5:]) not in split_info['train']:
            continue
        for cap in caps:
            for i, word_id in enumerate(cap[:-1]):
                if word_id not in word_next.keys():
                    word_next[word_id] = {}

                word_next[word_id][cap[i+1]] = word_next[word_id].get(cap[i+1], 0) + 1
    next_info = {'frequency': {}, 'word': {}}

    for key in word_next.keys():
        next_info['frequency'][key] = []
        next_info['word'][key] = []
        Sum = 0
        for w, fre in word_next[key].items():
            next_info['word'][key].append(w)
            Sum += fre
        for w, fre in word_next[key].items():
            next_info['frequency'][key].append(fre / Sum)

        preSum = 0
        for i in range(len(next_info['frequency'][key])):
            preSum += next_info['frequency'][key][i]
            next_info['frequency'][key][i] = preSum
    return next_info

def main(params):
    ori_info = pickle.load(open(params['ori_corpus'], 'rb'))['info'] if params['transform_distillation_corpus'] else None

    if ori_info is not None:
        split_info = ori_info['split']
    else:
        split_info = {
            'train': [i for i in range(1200)],
            'validate': [i for i in range(1200, 1300)],
            'test': [i for i in range(1300, 1970)]
        }
    
    split_info['trainval'] = split_info['train'] + split_info['validate']
    split_info['all'] = split_info['trainval'] + split_info['test']

    video_caption = defaultdict(list)
    trainingset_video_caption = {}

    refs = pickle.load(open(params['refs'], 'rb'))

    most_frequent_noun_verb = None
    attribute_mapping = []

    if params['my_noun_verb']:
        most_frequent_noun_verb = {}
        most_frequent = params['most_frequent']

    for vid in refs.keys():
        num = int(vid[5:]) # e.g. 'video999', num = 999
        for item in refs[vid]:
            cap = item['caption']
            video_caption[vid].append(cap)
            if params['my_noun_verb'] and num in split_info['train']:
                res = nltk.pos_tag(cap.split(' '))
                for w, t in res:
                    if my_mapping[t] in ['NOUN', 'VERB'] and w not in ['is', 'are']:
                        most_frequent_noun_verb[w] = most_frequent_noun_verb.get(w, 0) + 1

        if num in split_info['train']:
            trainingset_video_caption[vid] = video_caption[vid]

    print('Creating vocabulary')
    vocab = build_vocab(trainingset_video_caption, count_thr=params['word_count_threshold'])
    #pickle.dump({w: i for i, w in enumerate(vocab)}, open('/home/yangbang/VideoCaptioning/ar_test.pkl', 'wb'))

    itow = {i + 5: w for i, w in enumerate(vocab)}
    itow[Constants.PAD] = Constants.PAD_WORD
    itow[Constants.UNK] = Constants.UNK_WORD
    itow[Constants.BOS] = Constants.BOS_WORD
    itow[Constants.EOS] = Constants.EOS_WORD
    itow[Constants.MASK] = Constants.MASK_WORD
    wtoi = {w: i for i, w in itow.items()}  # inverse table

    if params['my_noun_verb']:
        most_frequent_noun_verb = sorted(most_frequent_noun_verb.items(), key=lambda d:d[1], reverse = True)
        most_frequent_noun_verb = [item[0] for item in most_frequent_noun_verb]
        most_frequent_noun_verb = most_frequent_noun_verb[:most_frequent]
        for w in most_frequent_noun_verb:
            attribute_mapping.append(wtoi[w])


    ttoi = {}
    ttoi[Constants.PAD_WORD] = Constants.PAD
    ttoi[Constants.UNK_WORD] = Constants.UNK
    ttoi[Constants.BOS_WORD] = Constants.BOS
    ttoi[Constants.EOS_WORD] = Constants.EOS
    ttoi[Constants.MASK_WORD] = Constants.MASK
    tag_start_i = 5

    captions = defaultdict(list)
    pos_tags = defaultdict(list)
    length_info = {}
    max_length = 100
    nlp = spacy.load('en_core_web_sm')
    for vid, caps in tqdm(video_caption.items()):
        length_info[vid] = [0] * max_length

        #print(caps)
        #doc = nlp.pipe(caps, n_threads=16)
        for j, cap in enumerate(caps):
        #for item in doc:
            #cap = [token.text for token in item]
            cap = [w.lower() for w in cap.split()]
            #while check(cap): pass

            #print(len(cap))
            length_info[vid][min(len(cap), max_length-1)] += 1

            tag_res = nltk.pos_tag(cap)
            #doc = nlp(' '.join(cap))
            #tag_res = [token.pos_ for token in item]



            caption_id = [Constants.BOS]
            tagging_id = [Constants.BOS]
            for w, t in zip(cap, tag_res):
                assert t[0] == w
                tag = mapping_tag(t[1], w)
                #tag = t

                if w in wtoi.keys():
                    caption_id += [wtoi[w]]
                else:
                    caption_id += [Constants.UNK]
                    #tagging_id += [Constants.UNK]
                
                if tag not in ttoi.keys():
                    ttoi[tag] = tag_start_i
                    tag_start_i += 1
                tagging_id += [ttoi[tag]]

            caption_id += [Constants.EOS]
            tagging_id += [Constants.EOS]

            captions[vid].append(caption_id)
            pos_tags[vid].append(tagging_id)

    
    itot = {i: t for t, i in ttoi.items()}
    print(itot)

    dump_data = {
        'info': {
            'itow': itow,                       # id to word
            'itop': itot,                       # id to POS tag
            'itoa': get_tag_info(video_caption, my_data=most_frequent_noun_verb),# id to attribute
            'length_info': length_info,         # id to length info
            'split': split_info,                # {'train': [...], 'validate': [...], 'test': [...]}
            'next_info': get_next_info(captions, split_info),
            'attribute_mapping': attribute_mapping
        },
        'captions': captions,
        'pos_tags': pos_tags,
    }

    if ori_info is not None:
        itoc = ori_info.get('itoc', None)
        split_category = ori_info.get('split_category', None)

        if itoc is not None: 
            dump_data['info']['itoc'] = itoc
        if split_category is not None: 
            dump_data['info']['split_category'] = split_category

    pickle.dump(dump_data, open(params["corpus"], 'wb'))
    print(attribute_mapping)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--base_pth', type=str, default='/home/yangbang/VideoCaptioning/Youtube2Text/')
    parser.add_argument('-wct', '--word_count_threshold', default=0, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    parser.add_argument('-tdc', '--transform_distillation_corpus', type=str, default='')
    parser.add_argument('-ori_wct', type=int, default=0)
    parser.add_argument('-topk', type=int, default=5)

    parser.add_argument('-mnv', '--my_noun_verb', default=False, action='store_true')
    parser.add_argument('-mf', '--most_frequent', type=int, default=1000)


    args = parser.parse_args()

    if args.transform_distillation_corpus:
        args.base_pth = args.base_pth.replace('Youtube2Text', args.transform_distillation_corpus)
        args.refs = os.path.join(args.base_pth, 'arvc%d_refs.pkl' % args.topk)
        args.corpus = os.path.join(args.base_pth, 'info_corpus_%d_%d.pkl' % (args.ori_wct, args.topk))
        args.ori_corpus = os.path.join(args.base_pth, 'info_corpus_%d.pkl' % args.ori_wct)

        assert os.path.exists(args.base_pth)
        assert os.path.exists(args.ori_corpus)

    else:
        if not os.path.exists(args.base_pth):
            os.makedirs(args.base_pth)

        args.refs = os.path.join(args.base_pth, 'msvd_refs.pkl')
        args.corpus = os.path.join(args.base_pth, 'info_corpus_%d.pkl' % (args.word_count_threshold))
        if not os.path.exists(args.refs):
            url = "https://github.com/ybCliff/VideoCaptioning/releases/download/1.0/msvd_refs.pkl"
            wget.download(url, out=args.refs)
    
    params = vars(args)  # convert to ordinary dict
    main(params)

'''
python msvd_prepross.py -tdc MSRVTT -ori_wct 2 -top_k 10
'''