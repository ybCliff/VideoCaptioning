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
import pickle
from collections import defaultdict
import spacy

train = '/home/yangbang/VideoCaptioning/VATEX/vatex_training_v1.0.json'
validate = '/home/yangbang/VideoCaptioning/VATEX/vatex_validation_v1.0.json'
test = '/home/yangbang/VideoCaptioning/VATEX/vatex_public_test_without_annotations.json'

def build_vocab(trainvids, count_thr):
    # count up the number of words
    counts = {}
    for vid, caps in trainvids.items():
        for cap in caps['captions']:
            cap = cap.split(' ')
            for w in cap:
                counts[w] = counts.get(w, 0) + 1

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


def load_data(file, id_to_vid, split_info, key, idx, videodatainfo, no_sent=False):
    video_caption = {}
    data = json.load(open(file))
    tqdm.write('===> Loading %s data' % key)
    for item in tqdm(data):
        vid = 'video%d' % idx
        split_info[key].append(idx)
        id_to_vid[item['videoID']] = vid
        if not no_sent:
            video_caption[vid] = {'captions': []}
            for sent in item['enCap']:
                tokens = nltk.word_tokenize(sent)
                caption = []
                for char in tokens:
                    if char not in string.punctuation:
                        caption.append(char.lower())
                caption = ' '.join(caption)
                video_caption[vid]['captions'].append(caption)
                videodatainfo.append({'video_id': vid, 'caption': caption})
        idx += 1
    return video_caption, idx

def pre_process_vatex():
    sentences = []
    videodatainfo = {'sentences': []}
    idx = 0
    split_info = {'train': [], 'validate': [], 'test': [], 'trainval': []}
    id_to_vid = {}

    trainingset_video_caption, idx = load_data(train, id_to_vid, split_info, 'train', idx, videodatainfo['sentences'], no_sent=False)
    video_caption, idx = load_data(validate, id_to_vid, split_info, 'validate', idx, videodatainfo['sentences'], no_sent=False)
    _, idx = load_data(test, id_to_vid, split_info, 'test', idx, videodatainfo['sentences'], no_sent=True)
    video_caption.update(trainingset_video_caption)

    return split_info, id_to_vid, trainingset_video_caption, video_caption, videodatainfo

def read_vocab_file(vocab_file):
    label_num = 0
    vocab = []
    with open(vocab_file, 'r') as f:
        for line in f:
            line = line.strip()
            vocab.append(line)
            label_num += 1
    return vocab, label_num

def get_tag_info(video_caption):
    f400, label_num = read_vocab_file("/home/yangbang/VideoCaptioning/most_frequent_400.txt")
    assert label_num == 400
    tag_info = {}
    for vid in video_caption.keys():
        caps = video_caption[vid]['final_captions']
        tag_info[vid] = [0] * label_num
        for cap in caps:
            for w in cap[1:-1]:
                if w in f400:
                    pos = f400.index(w)
                    tag_info[vid][pos] = 1
        #print(tag_info[vid])
    return tag_info

def mapping_tag(tag, w):
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

def main(params):
    split_info, id_to_vid, trainingset_video_caption, video_caption, videodatainfo = pre_process_vatex()
    split_info['trainval'] = split_info['train'] + split_info['validate']
    split_info['all'] = split_info['trainval'] + split_info['test']

    # create the vocab
    vocab = build_vocab(trainingset_video_caption, params['word_count_threshold'])
    itow = {i + 5: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 5 for i, w in enumerate(vocab)}  # inverse table

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
    nlp = spacy.load('en_core_web_sm')
    for vid, caps in tqdm(video_caption.items()):
        caps = caps['captions']
        video_caption[vid]['final_captions'] = []
        video_caption[vid]['final_captions_id'] = []
        video_caption[vid]['tagging_id'] = []

        doc = nlp.pipe(caps, n_threads=16)
        #for cap in caps:
        for item in doc:
            cap = [token.text for token in item]
            tag_res = [token.pos_ for token in item]

        
            #cap = cap.split(' ')
            max_length = max(len(cap), max_length)
            caption = [Constants.BOS_WORD]
            caption_id = [Constants.BOS]
            tagging_id = [Constants.BOS]
            
            #tag_res = nltk.pos_tag(cap)

            for w, t in zip(cap, tag_res):
                #assert t[0] == w
                #tag = mapping_tag(t[1], w)
                tag = t

                if w in wtoi.keys():
                    caption += [w]
                    caption_id += [wtoi[w]]
                    if tag not in ttoi.keys():
                        ttoi[tag] = tag_start_i
                        tag_start_i += 1
                    tagging_id += [ttoi[tag]]
                else:
                    caption += [Constants.UNK_WORD]
                    caption_id += [Constants.UNK]
                    tagging_id += [Constants.UNK]
            caption += [Constants.EOS_WORD]
            caption_id += [Constants.EOS]
            tagging_id += [Constants.EOS]

            video_caption[vid]['final_captions'].append(caption)
            video_caption[vid]['final_captions_id'].append(caption_id)
            video_caption[vid]['tagging_id'].append(tagging_id)

    itot = {i: t for t, i in ttoi.items()}

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
    out['id_to_vid'] = id_to_vid
    '''
    json.dump(out, open(params['info_json'], 'w'))
    json.dump(videodatainfo, open(params['videodatainfo'], 'w'))
    json.dump(video_caption, open(params['caption_json'], 'w'))
    json.dump(next_info, open(params["next_info"], 'w'))

    all_cap = {}
    all_cap['ix_to_word'] = itow
    all_cap['word_to_ix'] = wtoi
    all_cap['train'] = []
    for i in range(len(split_info['train'])):
        vid = 'video%d'%i
        length = len(video_caption[vid]['final_captions'])
        for j in range(length):
            caption = video_caption[vid]['final_captions'][j]
            caption_id = video_caption[vid]['final_captions_id'][j]
            all_cap['train'].append({'vid': vid, 'caption': caption, 'caption_id': caption_id})
    json.dump(all_cap, open(params["all_caption"], 'w'))
    '''

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
                'itop': itot,                       # id to POS tag
                'itoa': get_tag_info(video_caption),# id to attribute
                'length_info': length_info,         # id to length info
                'split': split_info,                # {'train': [...], 'validate': [...], 'test': [...]}
                'next_info': next_info,
                'id_to_vid': id_to_vid
            },
            'captions': captions,
            'pos_tags': pos_tags,
            #'references': references,
        }, 
        open(params["corpus"], 'wb')
    )

    refs = defaultdict(list)
    for item in videodatainfo['sentences']:
        vid = item['video_id']
        cap = item['caption']
        refs[vid].append({'image_id': vid, 'cap_id': len(refs[vid]), 'caption': cap})

    pickle.dump(
            refs,
            open(params["refs"], 'wb')
        )

    print(len(vocab))
    print(itot)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--base_pth', type=str, default='../VATEX/')
    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('-wct', '--word_count_threshold', default=2, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    if not os.path.exists(args.base_pth):
        os.makedirs(args.base_pth)

    args.videodatainfo = os.path.join(args.base_pth, 'videodatainfo.json')
    args.info_json = os.path.join(args.base_pth, '%sinfo_pad_mask_%d.json' % (args.prefix, args.word_count_threshold))
    args.caption_json = os.path.join(args.base_pth, '%scaption_pad_mask_%d.json' % (args.prefix, args.word_count_threshold))
    args.all_caption = os.path.join(args.base_pth, '%sall_caption_pad_mask_%d.json' % (args.prefix, args.word_count_threshold))
    args.next_info = os.path.join(args.base_pth, '%snext_info_pad_mask_%d.json' % (args.prefix, args.word_count_threshold))

    args.corpus = os.path.join(args.base_pth, '%sinfo_corpus_%d.pkl' % (args.prefix, args.word_count_threshold))
    args.refs = os.path.join(args.base_pth, 'vatex_refs.pkl')
    params = vars(args)  # convert to ordinary dict
    main(params)
