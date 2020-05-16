import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import json
import os
import argparse
import torch
import shutil
from torch.utils.data import DataLoader
from dataloader import VideoDataset as VD2
from STloader import VideoDataset as VD
from misc.cocoeval import suppress_stdout_stderr, COCOScorer
import numpy as np
from pandas.io.json import json_normalize
from misc.run import get_model, get_loader, get_forword_results
import misc.utils as utils
from misc.beam_search import ensemble_beam_search
import cv2
import sys

def change(mylist):
    mylist = [round(item * 100, 2) for item in mylist]
    return ',  '.join([str(i) for i in mylist])

def idx_to_frameid(path, idx):
    length = len(os.listdir(path))
    bound = [int(i) for i in np.linspace(0, length, 61)]
    index = []
    for i in range(60):
        index.append((bound[i] + bound[i+1])//2)

    return index[idx]+1

def show_two_stream_att(opt, model, data, vocab):
    results = get_forword_results(opt, model, data, mode='inference', modality=opt['modality'])
    seq_preds, seq_probs = results['seq_preds'], results['seq_probs']
    decoder_att_table = [results['decoder_att_table_reasoning'], results['decoder_att_table_guidance']]
    score_relationship = results['score_relationship']

    print(decoder_att_table[0].shape, decoder_att_table[1].shape)

    sents = utils.decode_sequence(vocab, seq_preds.reshape(-1, opt['max_len']-1))
    for i in range(len(sents)):
        logprob = seq_probs[0, i].sum()
        score1 = logprob / (len(sents[i].split(' ')) + 1)**opt['beam_alpha']
        score2 = logprob / (len(sents[i].split(' ')) + 1)
        
        print("%40s\t%.3f\t%.3f\t%.3f\t%d" % (sents[i], logprob, score1, score2, len(sents[i].split(' '))))
    
    idx = data['random_idx'].numpy().tolist()[0]
    vid = data['idx'].numpy().tolist()[0]
    print(idx, vid)
    assert(len(idx) == opt['n_frames'])

    sent = sents[0].split(' ')
    for i in range(len(sent)):
        print(score_relationship[0, i])


    n_frames = opt['n_frames']
    assert len(idx) == n_frames
    f, axarr = plt.subplots(3, n_frames)
    for i in range(n_frames):
        ax = axarr[0, i]
        frames_pth = os.path.join(opt['frames_pth'], 'video%d' % vid)
        frameid = idx_to_frameid(frames_pth, idx[i])
        pth = os.path.join(frames_pth, 'image_%05d.jpg'%frameid)
        img = cv2.imread(pth)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_yticks([])
        ax.set_xticks([])

    select_word = [1, 3, 5]
    color = ['r', 'g', 'b', 'y', 'm']
    for i in range(2):
        att_tabel = decoder_att_table[i]
        for j in range(n_frames):
            ax = axarr[i+1, j]
            maxValue = []
            maxValue.append(all_att_tabel[k][0, 0].max())
            maxValue = max(maxValue)
            maxValue = int(100 * maxValue) + 5

            width = 0.8

            att = []
            for k in range(len(opt)):
                att.append(all_att_tabel[k][0, 0, i, j])
            num_list = [round(100 * item, 1) for item in att]

            rects = ax.bar(range(len(num_list)), num_list, width, color='rgbym')
            
            ax.set_ylim(ymax=maxValue, ymin=0)
            ax.set_yticks([])
            ax.set_xticks([])
            if not j:
                #ax.set_ylabel(sent[i])
                #ax.text(0, 0, sent[i])
                left, width = 0, 1
                bottom, height = 0, 1
                right = left + width
                top = bottom + height
                ax.text(-0.1, 0.5, sent[i],
                horizontalalignment='right',
                verticalalignment='center',
                rotation=0,
                fontsize=15,
                transform=ax.transAxes)
                #ax.get_ylabel().set_rotation(90)
            
            #ax.axis('off')
            ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)
            #ax.spines['bottom'].set_visible(False)
            #ax.spines['left'].set_visible(False)

            for rect in rects:
                height = rect.get_height()
                if height:
                    ax.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
    plt.show()
    

def show2():
    n_frames = 8
    f, axarr = plt.subplots(3, n_frames)
    for i in range(n_frames):
        ax = axarr[0, i]
        ax.set_yticks([])
        ax.set_xticks([])
        x = np.arange(-10, 11, 1)  #形成一个数组，第三个参数表示步长，#start,end,step
        y = x ** 2

        ax.plot(x, y)
        ax.text(3, 10, "function: y = x * x", size = 10, alpha = 0.2)
    
    plt.subplots_adjust(left=None, bottom=0.5, right=None, top=None, wspace=0, hspace=None) 
    f.text(0.2, 0.4, "a man is palying the piano", size = 15)
    
    plt.show()

def show(opt, model, data, vocab):
    all_sents = []
    all_att_tabel = []
    for j in range(len(opt)):
        results = get_forword_results(opt[j], model[j], data, mode='inference', modality=opt[j]['modality'])
        seq_preds = results['seq_preds']
        seq_probs = results['seq_probs']
        decoder_att_table = results['decoder_att_table']
        print(decoder_att_table.shape)
        sents = utils.decode_sequence(vocab, seq_preds.reshape(-1, opt[j]['max_len']-1))
        for i in range(len(sents)):
            logprob = seq_probs[0, i].sum()
            score1 = logprob / (len(sents[i].split(' ')) + 1)**opt[j]['beam_alpha']
            score2 = logprob / (len(sents[i].split(' ')) + 1)
            
            print("%40s\t%.3f\t%.3f\t%.3f\t%d" % (sents[i], logprob, score1, score2, len(sents[i].split(' '))))
        all_sents.append(sents)
        all_att_tabel.append(decoder_att_table)

    
    idx = data['random_idx'].numpy().tolist()[0]
    vid = data['idx'].numpy().tolist()[0]
    print(idx, vid)
    assert(len(idx) == opt[0]['n_frames'])
    
    jud = True
    print(all_sents[0][0])
    for i in range(1, len(opt)):
        if all_sents[i][0] != all_sents[i-1][0]:
            jud = False
        print(all_sents[i][0])

    if jud:
        sents = all_sents[0]
        sent = sents[0].split(' ')
        n_frames = opt[0]['n_frames']
        f, axarr = plt.subplots(1 + len(sent), n_frames)
        for i in range(n_frames):
            ax = axarr[0, i]
            frames_pth = os.path.join(opt[0]['frames_pth'], 'video%d' % vid)
            frameid = idx_to_frameid(frames_pth, idx[i])
            pth = os.path.join(frames_pth, 'image_%05d.jpg'%frameid)
            img = cv2.imread(pth)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_yticks([])
            ax.set_xticks([])


        for i in range(len(sent)):
            for j in range(n_frames):
                ax = axarr[i+1, j]
                maxValue = []
                for k in range(len(opt)):
                    maxValue.append(all_att_tabel[k][0, 0].max())
                maxValue = max(maxValue)
                maxValue = int(100 * maxValue) + 5

                width = 0.8

                att = []
                for k in range(len(opt)):
                    att.append(all_att_tabel[k][0, 0, i, j])
                num_list = [round(100 * item, 1) for item in att]

                rects = ax.bar(range(len(num_list)), num_list, width, color='rgbym')
                
                ax.set_ylim(ymax=maxValue, ymin=0)
                ax.set_yticks([])
                ax.set_xticks([])
                if not j:
                    #ax.set_ylabel(sent[i])
                    #ax.text(0, 0, sent[i])
                    left, width = 0, 1
                    bottom, height = 0, 1
                    right = left + width
                    top = bottom + height
                    ax.text(-0.1, 0.5, sent[i],
                    horizontalalignment='right',
                    verticalalignment='center',
                    rotation=0,
                    fontsize=15,
                    transform=ax.transAxes)
                    #ax.get_ylabel().set_rotation(90)
                
                #ax.axis('off')
                ax.spines['top'].set_visible(False)
                #ax.spines['right'].set_visible(False)
                #ax.spines['bottom'].set_visible(False)
                #ax.spines['left'].set_visible(False)

                for rect in rects:
                    height = rect.get_height()
                    if height:
                        ax.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
        plt.show()
    
def get_forward_results_ensemble(option, opt, model, data, return_softmax_score=False, use_score_relationship=True):
    encoder_outputs, decoder_hiddens, additional_feats = [], [], []
    rnn, att, embedding, out = [], [], [], []

    vid_feats = data['fc_feats'].cuda()
    c3d_feats = data['c3d_feats'].cuda()
    acoustic_feats = data['acoustic_feats'].cuda()
    discriminative_feats = data['discriminative_feats'].cuda()
    category = data['category'].cuda() if option['use_ltm'] else None
    enhance_feats = None

    with torch.no_grad():
        for i in range(len(model)):
            eo, eh, add = model[i].encoder_feats(vid_feats, discriminative_feats, c3d_feats, acoustic_feats, enhance_feats, opt[i])
            #print(eo[0][0, 0, :5], vid_feats[0, 0, :10])
            #print(eh[0, 0, :10])
            additional_feats.append(add)
            tmp = []
            for name, module in model[i].decoder.named_children():
                if 'rnn' in name:
                    tmp.append(module)
            rnn.append(tmp if len(tmp) > 1 else tmp[0])
            dh = model[i].decoder.init_hidden(eh)

            encoder_outputs.append(eo)
            decoder_hiddens.append(dh if len(tmp) == 1 else [None, dh])

            att.append(model[i].decoder.att)
            embedding.append(model[i].decoder.embedding)
            out.append(model[i].decoder.out)

        return ensemble_beam_search(
                option, rnn, att, 
                decoder_hiddens, embedding, out, 
                encoder_outputs, model[0].decoder.get_log_prob, 
                additional_feats=additional_feats, 
                enhance_feats=enhance_feats, 
                decoder_bn=None, 
                additional_bn=None, 
                category=category, 
                decoder_type='gru',
                return_softmax_score=return_softmax_score,
                use_score_relationship=use_score_relationship
            )

def myPrint(content, file=''):
    if file:
        f = open(file, 'a')
        f.write(content + '\n')
        f.close()
    print(content)

def DM3L_show(option, opt, model, data, vocab, examples_pth=''):
    all_sents = []
    all_att_tabel = []
    for j in range(len(opt) + 1):
        if j == len(opt):
            seq_probs, seq_preds, decoder_att_table, score_relationship = get_forward_results_ensemble(option, opt[1:], model[1:], data)
        else:
            results = get_forword_results(opt[j], model[j], data, mode='inference', modality=opt[j]['modality'])
            seq_preds = results['seq_preds']
            seq_probs = results['seq_probs']
            decoder_att_table = results['decoder_att_table']

        sents = utils.decode_sequence(vocab, seq_preds.reshape(-1, option['max_len']-1))
        for i in range(len(sents)):
            logprob = seq_probs[0, i].sum()
            score1 = logprob / (len(sents[i].split(' ')) + 1)**option['beam_alpha']
            score2 = logprob / (len(sents[i].split(' ')) + 1)
            print("%40s\t%.3f\t%.3f\t%.3f\t%d" % (sents[i], logprob, score1, score2, len(sents[i].split(' '))))
            if j == len(opt):
                for k in range(len(sents[i].split(' '))):
                    print(score_relationship[i][k])

        print('------------------------')
        all_sents.append(sents)
        if isinstance(decoder_att_table, list):
            for item in decoder_att_table:
                all_att_tabel.append(item[np.newaxis, :, :, :])
        else:
            all_att_tabel.append(decoder_att_table)

    for item in all_att_tabel:
        print(item.shape)

    #tmp_idx = 0
    #print(all_att_tabel[0][0, 0, tmp_idx, :])
    #print(all_att_tabel[2][0, 0, tmp_idx, :])
    #print('----')
    #print(all_att_tabel[1][0, 0, tmp_idx, :])
    #print(all_att_tabel[3][0, 0, tmp_idx, :])

    idx = data['random_idx'].numpy().tolist()[0]
    vid = data['idx'].numpy().tolist()[0]
    print(idx, vid)
    assert(len(idx) == opt[0]['n_frames'])
    if examples_pth:
        examples_pth = os.path.join(examples_pth, 'video%d'%vid)
        txt_pth = os.path.join(examples_pth, 'captions.txt')
        if not os.path.exists(examples_pth):
            os.makedirs(examples_pth)

    if option['dataset'] == 'MSRVTT':
        caption = json.load(open("/home/yangbang/VideoCaptioning/MSRVTT/caption_2.json"))
        caption2 = json.load(open("/home/yangbang/VideoCaptioning/MDH-S2VTAtt/ensemble_results/test/MSRVTT_zo(ica0)_zi(ica1002)/caption_results.json"))['predictions']
    else:
        caption = json.load(open("/home/yangbang/VideoCaptioning/Youtube2Text/caption_0.json"))
    for i in range(len(caption['video%d'%vid]['captions'])):
        myPrint(caption['video%d'%vid]['captions'][i], file=txt_pth if examples_pth else '')
    myPrint('-----------', file=txt_pth if examples_pth else '')
    for i in range(len(all_sents)):
        myPrint(all_sents[i][0], file=txt_pth if examples_pth else '')
    print(caption2['video%d'%vid][0]['caption'])
    

    if len(option['visualize']):
        sents = all_sents[-1]
        att_tabel = all_att_tabel[3:]
        sent = sents[0].split(' ')
        n_frames = opt[0]['n_frames']
        f, axarr = plt.subplots(1 + 2, n_frames)
        for i in range(n_frames):
            ax = axarr[0, i]
            frames_pth = os.path.join(opt[0]['frames_pth'], 'video%d' % vid)
            frameid = idx_to_frameid(frames_pth, idx[i])
            print(frameid)
            pth = os.path.join(frames_pth, 'image_%05d.jpg'%frameid)
            if examples_pth:
                shutil.copy(pth, os.path.join(examples_pth, 'image_%05d.jpg'%frameid))

            img = cv2.imread(pth)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(str(vid))

        select_word = option['visualize']
        for item in select_word:
            print(sent[item], score_relationship[0][item])

        if examples_pth:
            for i in range(len(sent)):
                myPrint(str(score_relationship[0][i]), txt_pth)


        for i, mode in enumerate(['reasoning', 'guidance']):
            maxValue = []
            for k in select_word:
                maxValue.append(att_tabel[i][0, 0, k].max())
            maxValue = max(maxValue)
            maxValue = int(100 * maxValue) + 5

            for j in range(n_frames):
                ax = axarr[1 + i, j]
                att = []
                for k in range(len(select_word)):
                    att.append(att_tabel[i][0, 0, select_word[k], j])
                num_list = [round(100 * item, 1) for item in att]
                num_list.insert(0, 0)
                num_list.append(0)
                rects = ax.bar(range(len(num_list)), num_list, 0.8, color='mgybry')
                ax.set_ylim(ymax=maxValue, ymin=0)
                ax.set_yticks([])
                ax.set_xticks([])
                if not j:
                    #ax.set_ylabel(sent[i])
                    #ax.text(0, 0, sent[i])
                    left, width = 0, 1
                    bottom, height = 0, 1
                    right = left + width
                    top = bottom + height
                    ax.text(-0.1, 0.5, mode,
                    horizontalalignment='right',
                    verticalalignment='center',
                    rotation=0,
                    fontsize=8,
                    transform=ax.transAxes)
                    #ax.get_ylabel().set_rotation(90)
                
                #ax.axis('off')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

                for rect in rects:
                    height = rect.get_height()
                    if height:
                        ax.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')

        plt.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=0, hspace=None)
        f.text(0.2, 0.30, caption['video%d'%vid]['captions'][0])
        f.text(0.2, 0.24, all_sents[0][0])
        f.text(0.2, 0.18, all_sents[1][0])
        f.text(0.2, 0.12, all_sents[2][0])
        f.text(0.2, 0.06, all_sents[3][0])
        if not examples_pth:
            plt.show()
        else:
            plt.savefig(os.path.join(examples_pth, 'demo.png'))

def IPE_show(option, opt, model, data, vocab, examples_pth=''):
    all_sents = []
    all_att_tabel = []
    for j in range(len(opt)):
        results = get_forword_results(opt[j], model[j], data, mode='inference', modality=opt[j]['modality'])
        seq_preds = results['seq_preds']
        seq_probs = results['seq_probs']
        decoder_att_table = results['decoder_att_table']

        sents = utils.decode_sequence(vocab, seq_preds.reshape(-1, option['max_len']-1))
        for i in range(len(sents)):
            logprob = seq_probs[0, i].sum()
            score1 = logprob / (len(sents[i].split(' ')) + 1)**option['beam_alpha']
            score2 = logprob / (len(sents[i].split(' ')) + 1)
            print("%40s\t%.3f\t%.3f\t%.3f\t%d" % (sents[i], logprob, score1, score2, len(sents[i].split(' '))))

        print('------------------------')
        all_sents.append(sents)
        all_att_tabel.append(decoder_att_table)

    for item in all_att_tabel:
        print(item.shape)

    idx = data['random_idx'].numpy().tolist()[0]
    vid = data['idx'].numpy().tolist()[0]
    print(idx, vid)
    assert(len(idx) == opt[0]['n_frames'])
    if examples_pth:
        examples_pth = os.path.join(examples_pth, 'video%d'%vid)
        txt_pth = os.path.join(examples_pth, 'captions.txt')
        if not os.path.exists(examples_pth):
            os.makedirs(examples_pth)

    if option['dataset'] == 'MSRVTT':
        caption = json.load(open("/home/yangbang/VideoCaptioning/MSRVTT/caption_2.json"))
        caption2 = json.load(open("/home/yangbang/VideoCaptioning/MDH-S2VTAtt/ensemble_results/test/MSRVTT_zi(ica1002)/caption_results.json"))['predictions']
    else:
        caption = json.load(open("/home/yangbang/VideoCaptioning/Youtube2Text/caption_0.json"))
    for i in range(len(caption['video%d'%vid]['captions'])):
        myPrint(caption['video%d'%vid]['captions'][i], file=txt_pth if examples_pth else '')
    myPrint('-----------', file=txt_pth if examples_pth else '')
    for i in range(len(all_sents)):
        myPrint(all_sents[i][0], file=txt_pth if examples_pth else '')
    print(caption2['video%d'%vid][0]['caption'])
    

    if len(option['visualize']):
        sents = all_sents[-1]
        att_tabel = all_att_tabel[-1]
        sent = sents[0].split(' ')
        n_frames = opt[0]['n_frames']
        f, axarr = plt.subplots(1 + 1, n_frames)
        for i in range(n_frames):
            ax = axarr[0, i]
            frames_pth = os.path.join(opt[0]['frames_pth'], 'video%d' % vid)
            frameid = idx_to_frameid(frames_pth, idx[i])
            print(frameid)
            pth = os.path.join(frames_pth, 'image_%05d.jpg'%frameid)
            if examples_pth:
                shutil.copy(pth, os.path.join(examples_pth, 'image_%05d.jpg'%frameid))

            img = cv2.imread(pth)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(str(vid))

        select_word = option['visualize']


        maxValue = []
        for k in select_word:
            maxValue.append(att_tabel[0, 0, k].max())
        maxValue = max(maxValue)
        maxValue = int(100 * maxValue) + 5

        for j in range(n_frames):
            ax = axarr[1, j]
            att = []
            for k in range(len(select_word)):
                att.append(att_tabel[0, 0, select_word[k], j])
            num_list = [round(100 * item, 1) for item in att]
            num_list.insert(0, 0)
            num_list.append(0)
            rects = ax.bar(range(len(num_list)), num_list, 0.8, color='mgybry')
            ax.set_ylim(ymax=maxValue, ymin=0)
            ax.set_yticks([])
            ax.set_xticks([])
            
            #ax.axis('off')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            for rect in rects:
                height = rect.get_height()
                if height:
                    ax.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')

        plt.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=0, hspace=None)
        f.text(0.2, 0.30, caption['video%d'%vid]['captions'][0])
        f.text(0.2, 0.24, all_sents[0][0])
        f.text(0.2, 0.18, all_sents[1][0])
        f.text(0.2, 0.12, all_sents[2][0])
        if not examples_pth:
            plt.show()
        else:
            plt.savefig(os.path.join(examples_pth, 'demo.png'))

def main(option, opt):
    model_list = []
    for i in range(len(opt)):
        model = get_model(opt[i])
        print(model)
        checkpoint = torch.load(opt[i]['model_pth'], 'cpu')
        model.load_state_dict(checkpoint['state_dict'])
        if opt[i]['beam_alpha'] == 0.0:
            opt[i]['beam_alpha'] = checkpoint['beam_alpha']
            print('checkpoint beam_alpha: %f' % checkpoint['beam_alpha'])
        model.cuda()
        model.eval()
        model_list.append(model)

    loader, vocab, _ = get_loader(opt[0], 'test', False)
    loader = iter(loader)

    if option['save_example']:
        examples = sorted(option['examples'])
        examples = [item - 7009 for item in examples]
        index, itr = 0, 0
        while index < len(examples):
            data = loader.next()
            itr += 1
            if itr == examples[index]:
                index += 1
                #DM3L_show(option, opt, model_list, data, vocab, examples_pth=option['examples_pth'])
                IPE_show(option, opt, model_list, data, vocab, examples_pth=option['examples_pth'])
    else:
        if option['num'] > 7009:
            option['num'] -= 7009
        for i in range(option['num']):
            data = loader.next()
        #DM3L_show(option, opt, model_list, data, vocab)
        IPE_show(option, opt, model_list, data, vocab)

        
def find_vid_from_specific_gt(gt, dataset):
    if dataset == 'MSRVTT':
        caption = json.load(open("/home/yangbang/VideoCaptioning/MSRVTT/caption_2.json"))
        limit = 10000
    else:
        caption = json.load(open("/home/yangbang/VideoCaptioning/Youtube2Text/caption_0.json"))
        limit = 1970

    for i in range(limit):
        vid = i
        for cap in caption['video%d'%vid]['captions']:
            if cap == gt:
                print(vid, vid-7009)
                print(caption['video%d'%vid]['captions'])
                break

    

    
if __name__ == '__main__':
    #torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_pth', nargs='+', type=str, default=[
        #"/home/yangbang/VideoCaptioning/729save/MSRVTT/baseline/DFM_Model_linearlstm_icas/SR8_wMA0_wt2_SS1_pEc_together_100_70_noHI_wise_Avb256f260_EBN_ltm20/",
        #"/home/yangbang/VideoCaptioning/729save/MSRVTT/zo/DFM_Model_grulstm_icas/SR8_wMA0_wt2_SS1_pEc_100_70_noHI_wise_Avb256f260_EBN_ltm20/", 
        #"/home/yangbang/VideoCaptioning/729save/MSRVTT/zi/DFM_Model_grulstm_icas/SR8_wMA0_wt2_SS1_pEc_100_70_noHI_Avb256f260_EBN_ltm20_1002/",
        "/home/yangbang/VideoCaptioning/829save/MSRVTT/baseline/DFM_Model_linearlstm_icas/SR8wt2_GFlow_RDirect_pEc_SS1_together_100_70_Avb256f260_EBN_ltm20_1002/",
        "/home/yangbang/VideoCaptioning/829save/MSRVTT/baseline/DFM_Model_mslstmlstm_icas/SR8wt2_GFlow_RDirect_pEc_SS1_100_70_Avb256f260_EBN_ltm20_1002/",
        "/home/yangbang/VideoCaptioning/829save/MSRVTT/zi/DFM_Model_grulstm_icas/A_SR8wt2_GFlow_RDirect_pEc_SS1_100_70__Avb256f260_EBN_ltm20_1002/",


        ])
    parser.add_argument('--model_name', type=str, default=[
        #'0_0393_177633_180404_174457_184132_182915.pth.tar',
        #'0_0452_183134_191522_182142_191021_190094.pth.tar',
        #'0_0402_185147_190982_182617_191536_191737.pth.tar',
        '0_0474_176009_182540_173166_183347_181886.pth.tar',
        '0_0411_183381_194148_179707_188063_186484.pth.tar',
        '0_0453_185254_192254_183016_193590_192219.pth.tar',


        ])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('-bs', '--beam_size', type=int, default=5, help='used when sample_max = 1. Usually 2 or 3 works well.')
    parser.add_argument('-bc', '--beam_candidate', type=int, default=5)
    parser.add_argument('-ba', '--beam_alpha', type=float, default=0.0)
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--dataset', default='Youtube2Text', type=str)
    parser.add_argument('--frames_pth', type=str, default='/home/yangbang/VideoCaptioning/MSRVTT/all_frames/')
    parser.add_argument('--n_frames', type=int, default=8)
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--use_ltm', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', nargs='+', type=int, default=[])
    parser.add_argument('--find', default=False, action='store_true')
    
    parser.add_argument('-se', '--save_example', default=False, action='store_true')
    parser.add_argument('--examples', nargs='+', type=int, default=[
        #7034, 7061, 7064, 7077, 7087, 7120, 7123, 7129, 7140, 7166, 7184, 7190, 7247, 7299, 7476, 7481, 7525, 7607, 7675, 7701, 7955
        #7044, 7268, 7405, 7687,  7699, 7799, 7876, 7915, 7987, 7996, 8131, 8154, 8155, 8297, 8325, 8342, 8352, 8528, 8781, 8789, 8834, 9097, 9426, 9505, 9587, 9731, 9982
        7097, 7317, 7711, 7879, 8660, 8938, 9255, 9262, 9338, 9784
        ])
    parser.add_argument('--examples_pth', type=str, default='./qualitative_examples')

    args = parser.parse_args()
    args = vars((args))
    
    if not args['find']:
        assert args['dataset'] in ['Youtube2Text', 'MSRVTT']
        args['frames_pth'] = '/home/yangbang/VideoCaptioning/%s/all_frames/' % args['dataset']
        args['max_len'] = 20 if args['dataset'] == 'Youtube2Text' else 30

        length = len(args['model_name'])
        assert length == len(args['base_pth'])

        recover_opt = []
        model_pth = []
        for i in range(length):
            recover_opt.append(os.path.join(args['base_pth'][i], 'opt_info.json'))
            model_pth.append(os.path.join(args['base_pth'][i], 'best', args['model_name'][i]))

        opt_list = []
        for i in range(length):    
            opt = json.load(open(recover_opt[i]))
            for k, v in args.items():
                opt[k] = v

            opt['recover_opt'] = recover_opt[i]
            opt['model_pth'] = model_pth[i]
            opt_list.append(opt)
        main(args, opt_list)
    else:
        find_vid_from_specific_gt("a man cuts a sausage on a cutting board", 'MSRVTT')


