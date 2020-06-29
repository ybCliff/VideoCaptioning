''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import os,json
from misc.run import get_model, get_loader, run_eval, get_forword_results
import torch.nn.functional as F
from misc.logger import CsvLogger
import matplotlib.pyplot as plt
import numpy as np 
def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model_path', type=str, default="/home/yangbang/VideoCaptioning/1007save/MSRVTT/IPE_LSTM/AMIs_Seed0_EBN_WC20_SS1_100_70_PEami_tanh/")
    parser.add_argument('-model_name', default='0_0414_182243_190984_179509_189588_188303.pth.tar', type=str)
    #"/home/yangbang/VideoCaptioning/1007save/MSRVTT/IPE_LSTM/AMIs_Seed0_EBN_WC20_SS1_100_70_PEami_tanh/best/0_0434_182737_192768_178935_189619_188431.pth.tar"
    #"/home/yangbang/VideoCaptioning/1007save/MSRVTT/IPE_LSTM/AMIs_Seed0_EBN_WC20_SS1_100_70_PEami_tanh/best/0_0414_182243_190984_179509_189588_188303.pth.tar"
    #"/home/yangbang/VideoCaptioning/1007save/MSRVTT/IPE_LSTM/AMIs_Seed0_EBN_WC20_SS1_100_70_PEami_tanh/best/0_0331_183179_192238_176971_189766_188197.pth.tar"
    #""/home/yangbang/VideoCaptioning/1007save/MSRVTT/IPE_LSTM/Is_Seed0_EBN_WC20_SS1_100_70_PEi/best/0_0423_172539_174895_172912_180085_178539.pth.tar""
    #"/home/yangbang/VideoCaptioning/1007save/MSRVTT/IPE_LSTM/AIs_Seed0_EBN_WC20_SS1_100_70_PEai/best/0_0392_178565_186435_176093_185624_183662.pth.tar"


    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-beam_alpha', type=float, default=1.0)
    parser.add_argument('-batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('-topk', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-max_len', type=int, default=30)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-em', type=str, default='test')
    parser.add_argument('-print_sent', action='store_true')
    parser.add_argument('-func', type=int, default=1)
    parser.add_argument('-json_path', type=str, default='')
    parser.add_argument('-json_name', type=str, default='')
    parser.add_argument('-ns', default=False, action='store_true')
    parser.add_argument('-sv', default=False, action='store_true')
    parser.add_argument('-specific', default=-1, type=int)

    opt = parser.parse_args()
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')

    opt_pth = os.path.join(opt.model_path, 'opt_info.json')
    option = json.load(open(opt_pth, 'r'))
    option.update(vars(opt))
    print(option)

    model = get_model(option)
    checkpoint = torch.load(os.path.join(opt.model_path, 'best', opt.model_name))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    logger = CsvLogger(
        filepath=opt.model_path, 
        filename='category_wise.csv', 
        fieldsnames=['category', 'loss', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'Sum']
        )
    #for specific in range(20):
    loader = get_loader(option, mode=opt.em, print_info=False, specific=-1)
    vocab = loader.dataset.get_vocab()
    print(run_eval(option, model, None, loader, vocab, device))
    '''
        metric = run_eval(option, model, None, loader, vocab, device)
        print(metric)
        metric['category'] = specific
        logger.write(metric)
        '''
        
    '''
        model.eval()
        res = {}
        ln = torch.nn.LayerNorm(512).to(device)
        tanh = torch.tanh

        for j in range(len(option['preEncoder_modality'])):
            res[j] = [[] for _ in range(option['n_frames'])]
        for data in tqdm(loader, ncols=150, leave=False):
            with torch.no_grad():
                encoder_outputs, category, labels = get_forword_results(option, model, data, device=device, only_data=True)
                emb, enc_output = encoder_outputs['emb'], encoder_outputs['ori_eo']
                for i in range(option['n_frames']):
                    for j in range(len(emb)):
                        
                        #simi = F.pairwise_distance(ln(enc_output[:, i, :]), ln(emb[j][:, i, :]), p=2)
                        #simi = F.cosine_similarity(enc_output[:, i, :], tanh(emb[j][:, i, :]), dim=1)
                        simi = F.cosine_similarity(enc_output[:, i, :], emb[j][:, i, :], dim=1)
                        #simi = F.cosine_similarity(ln(enc_output[:, i, :]), ln(emb[j][:, i, :]), dim=1)
                        #torch.cosine_similarity(enc_output[:, i, :], tanh(emb[j][:, i, :]), dim=1)
                        res[j][i].append(simi)

        import math
        
        nf_ave = []
        for i in range(option['n_frames']):
            tmp = []
            for j in range(len(option['preEncoder_modality'])): 
                simi = torch.cat(res[j][i], dim=0).mean()
                tmp.append(torch.log(simi.abs_()))
            p = F.softmax(torch.stack(tmp, dim=0), dim=0)
            #print(i, p)
            nf_ave.append(p)
        nf_ave = torch.stack(nf_ave, dim=0).mean(0)
        print(specific, nf_ave)
        '''
    '''
        for data in tqdm(loader, ncols=150, leave=False):
            with torch.no_grad():
                encoder_outputs, category, labels = get_forword_results(option, model, data, device=device, only_data=True)
                emb, enc_output = encoder_outputs['emb'], encoder_outputs['ori_eo']

                p1 = plt.subplot(221)
                p1.bar([i for i in range(512)], enc_output[0, 0, :].cpu().numpy())

                p1 = plt.subplot(222)
                p1.bar([i for i in range(512)], emb[0][0, 0, :].cpu().numpy())

                p1 = plt.subplot(223)
                p1.bar([i for i in range(512)], emb[1][0, 0, :].cpu().numpy())

                p1 = plt.subplot(224)
                p1.bar([i for i in range(512)], emb[2][0, 0, :].cpu().numpy())

                plt.show()
            break
        '''

def test():
    
    hueHist = plt.subplot(121)
    num_bins = 3
    hueHist.bar([i for i in range(512)], np.random.rand(512))
    hueHist.set_title('hue frequency',fontsize=12,color='r')

     
    plt.show()
if __name__ == "__main__":
    main()
    #test()
