''' Handling the data io '''
import argparse
import models.Constants as Constants
import json
import numpy as np
from tqdm import tqdm
import string
import wget
import os
import nltk
from collections import defaultdict
import pickle
import spacy
url = "https://github.com/ybCliff/VideoCaptioning/releases/download/v1.0/videodatainfo_2016.json"

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
    counts = {}
    for vid, caps in trainvids.items():
        for cap in caps['captions']:
            # change all words to the lowercase
            cap = [w.lower() for w in cap.split()]
            # remove punctuation
            while check(cap): pass
            # count up the number of words
            for w in cap:
                counts[w] = counts.get(w, 0) + 1


    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]

    bad_count = sum(counts[w] for w in bad_words)

    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab)))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))

    return vocab

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



def main(params):
    sentences = json.load(open(params['input_json'], 'r'))['sentences']
    videos = json.load(open(params['input_json'], 'r'))['videos']

    split_info = {'train': [], 'validate': [], 'test': [], 'all': []}
    for v in videos:
        split_info[v['split']].append(int(v['id']))
    split_info['all'] = split_info['train'] + split_info['validate'] + split_info['test']

    video_caption = {}
    trainingset_video_caption = {}

    most_frequent_noun_verb = None

    if params['my_noun_verb']:
        most_frequent_noun_verb = {}
        most_frequent = params['most_frequent']

    for i in tqdm(sentences):
        if i['video_id'] not in video_caption.keys():
            video_caption[i['video_id']] = {'captions': []}
        video_caption[i['video_id']]['captions'].append(i['caption'].lower())


        if int(i['video_id'][5:]) in split_info['train']:
            if i['video_id'] not in trainingset_video_caption.keys():
                trainingset_video_caption[i['video_id']] = {'captions': []}
            trainingset_video_caption[i['video_id']]['captions'].append(i['caption'].lower())
    '''
    video_caption = {
        'video1': {
            'captions': ['a girl is singing', sent2, sent3],
            'final_captions': [['<bos>', 'a', 'girl', 'is', 'singing', '<eos>'], sent2, sent3],
            'final_captions_id': index version of 'final_captions'
        }
        'video2': ...
    }
    '''
    

    # create the vocab according to the training split
    vocab = build_vocab(trainingset_video_caption, params['word_count_threshold'])
    itow = {i + 5: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 5 for i, w in enumerate(vocab)}  # inverse dictionary

    wtoi[Constants.PAD_WORD] = Constants.PAD
    wtoi[Constants.UNK_WORD] = Constants.UNK
    wtoi[Constants.BOS_WORD] = Constants.BOS
    wtoi[Constants.EOS_WORD] = Constants.EOS
    wtoi[Constants.MASK_WORD] = Constants.MASK

    itow[Constants.PAD] = Constants.PAD_WORD
    itow[Constants.UNK] = Constants.UNK_WORD
    itow[Constants.BOS] = Constants.BOS_WORD
    itow[Constants.EOS] = Constants.EOS_WORD
    itow[Constants.MASK] = Constants.MASK_WORD

    ttoi = {}
    ttoi[Constants.PAD_WORD] = Constants.PAD
    ttoi[Constants.UNK_WORD] = Constants.UNK
    ttoi[Constants.BOS_WORD] = Constants.BOS
    ttoi[Constants.EOS_WORD] = Constants.EOS
    ttoi[Constants.MASK_WORD] = Constants.MASK
    tag_start_i = 5
    
    max_length = 0
    
    nlp = spacy.load("en_core_web_sm")
    for vid, caps in tqdm(video_caption.items()):
        caps = caps['captions']
        video_caption[vid]['final_captions'] = []
        video_caption[vid]['final_captions_id'] = []
        video_caption[vid]['tagging_id'] = []

        for cap in caps:
            cap = [w.lower() for w in cap.split()]
            while check(cap): pass

            tag_res = nltk.pos_tag(cap)
            #doc = nlp(' '.join(cap))
            #tag_res = [token.pos_ for token in doc]

            max_length = max(len(cap), max_length)
            caption = [Constants.BOS_WORD]

            tagging = [Constants.BOS_WORD]

            caption_id = [Constants.BOS]
            tagging_id = [Constants.BOS]
            for w, t in zip(cap, tag_res):
                assert t[0] == w
                tag = mapping_tag(t[1], w)

                if params['my_noun_verb']:
                    if tag in ['NOUN', 'VERB'] and w not in ['is', 'are']:
                        most_frequent_noun_verb[w] = most_frequent_noun_verb.get(w, 0) + 1

                #tag = t

                if w in wtoi.keys():
                    caption += [w]
                    caption_id += [wtoi[w]]
                    if tag not in ttoi.keys():
                        ttoi[tag] = tag_start_i
                        tag_start_i += 1
                    tagging_id += [ttoi[tag]]
                    tagging += [tag]
                else:
                    caption += [Constants.UNK_WORD]
                    caption_id += [Constants.UNK]
                    tagging_id += [Constants.UNK]
                    tagging += [Constants.UNK]
            caption += [Constants.EOS_WORD]
            caption_id += [Constants.EOS]
            tagging_id += [Constants.EOS]

            video_caption[vid]['final_captions'].append(caption)
            video_caption[vid]['final_captions_id'].append(caption_id)
            video_caption[vid]['tagging_id'].append(tagging_id)

    itot = {i: t for t, i in ttoi.items()}
    if params['my_noun_verb']:
        most_frequent_noun_verb = sorted(most_frequent_noun_verb.items(), key=lambda d:d[1], reverse = True)
        most_frequent_noun_verb = [item[0] for item in most_frequent_noun_verb]
        most_frequent_noun_verb = most_frequent_noun_verb[:most_frequent]


    length_info = {}
    for vid, caps in video_caption.items():
        caps = caps['final_captions']
        length_info[vid] = [0] * (max_length+1)
        for cap in caps:
            length = len(cap) - 2
            length_info[vid][length] += 1


    word_next = {}
    for vid, caps in video_caption.items():
        if int(vid[5:]) not in split_info['train']:
            continue
        caps = caps['final_captions_id']
        for cap in caps:
            for i, word_id in enumerate(cap[:-1]):
                if word_id == Constants.EOS:
                    break
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
        

    #print(next_info['frequency'])

    out = {}
    out['length_info'] = length_info
    out['ix_to_word'] = itow
    out['word_to_ix'] = wtoi
    out['videos'] = split_info
    out['ix_to_tag'] = itot
    out['tag_to_ix'] = ttoi
    out['tag_info'] = get_tag_info(video_caption)

    itoc = {}
    out['split_category'] = {'train': {}, 'validate': {}, 'test': {}, 'all': {}, 'no_train': {}}
    out['videos_category'] = {'train': {}, 'validate': {}, 'test': {}, 'all': {}, 'no_train': {}}
    
    count_category = np.zeros(params['num_class'])
    for i in videos:
        if int(i["category"]) not in out['split_category'][i['split']].keys():
            out['split_category'][i['split']][int(i["category"])] = []
        if int(i["category"]) not in out['split_category']['all'].keys():
            out['split_category']['all'][int(i["category"])] = []
        if i['split'] != 'train' and int(i["category"]) not in out['split_category']['no_train'].keys():
            out['split_category']['no_train'][int(i["category"])] = []

        out['split_category'][i['split']][int(i["category"])].append(int(i['id']))
        out['videos_category'][i['split']][int(i['id'])] = int(i["category"])
        
        out['split_category']['all'][int(i["category"])].append(int(i['id']))
        out['videos_category']['all'][int(i['id'])] = int(i["category"])

        if i['split'] != 'train':
            out['split_category']['no_train'][int(i["category"])].append(int(i['id']))
            out['videos_category']['no_train'][int(i['id'])] = int(i["category"])

        itoc[i["id"]] = i["category"]
        if i['split'] == 'train':
            count_category[i["category"]] += 1
    out["id_to_category"] = itoc
    out["count_category"] = count_category.tolist()
    #json.dump(out, open(params['info_json'], 'w'))
    #json.dump(video_caption, open(params['caption_json'], 'w'))
    #json.dump(next_info, open(params["next_info"], 'w'))


    captions = {}
    pos_tags = {}
    #references = defaultdict(list)
    for key in video_caption.keys():
        captions[key] = video_caption[key]['final_captions_id']
        pos_tags[key] = video_caption[key]['tagging_id']
        #for cap in video_caption[key]['captions']:
        #    references[key].append({'image_id': key, 'cap_id': len(references[key]), 'caption': cap})


    
    pickle.dump({
            'info': {
                'itow': itow,                       # id to word
                'itoc': itoc,                       # id to category
                'itop': itot,                       # id to POS tag
                'itoa': get_tag_info(video_caption, my_data=most_frequent_noun_verb),# id to attribute
                'length_info': length_info,         # id to length info
                'split': split_info,                # {'train': [...], 'validate': [...], 'test': [...]}
                'split_category': out['split_category'],
                'next_info': next_info
            },
            'captions': captions,
            'pos_tags': pos_tags,
            #'references': references,
        }, 
        open(params["corpus"], 'wb')
    )
    print(itot)
    print(len(vocab))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--base_pth', type=str, default='../MSRVTT/')
    parser.add_argument('--num_class', default=20, type=int)
    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('-wct', '--word_count_threshold', default=2, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('-mnv', '--my_noun_verb', default=False, action='store_true')
    parser.add_argument('-mf', '--most_frequent', type=int, default=1000)

    args = parser.parse_args()
    if not os.path.exists(args.base_pth):
        os.makedirs(args.base_pth)

    args.input_json = os.path.join(args.base_pth, '%svideodatainfo.json' % args.prefix)
    if not os.path.exists(args.input_json):
        wget.download(url, out=args.input_json)

    args.info_json = os.path.join(args.base_pth, '%sinfo_pad_mask_%d.json' % (args.prefix, args.word_count_threshold))
    args.caption_json = os.path.join(args.base_pth, '%scaption_pad_mask_%d.json' % (args.prefix, args.word_count_threshold))
    args.all_caption = os.path.join(args.base_pth, '%sall_caption_pad_mask_%d.json' % (args.prefix, args.word_count_threshold))
    args.next_info = os.path.join(args.base_pth, '%snext_info_pad_mask_%d.json' % (args.prefix, args.word_count_threshold))

    args.corpus = os.path.join(args.base_pth, '%sinfo_corpus_%d.pkl' % (args.prefix, args.word_count_threshold))
    params = vars(args)  # convert to ordinary dict
    main(params)
