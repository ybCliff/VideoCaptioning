import re
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os 
from misc.run import ensemble_evaluate
parser = argparse.ArgumentParser()



parser.add_argument('--ensemble_checkpoint_paths', nargs='+', type=str, default=[
    "/home/yangbang/VideoCaptioning/829save/MSRVTT/baseline/DFM_Model_grulstm_is/SR8wt2_GFlow_RDirect_SS1_100_70_EBN_ltm20_1002/",
    #"/home/yangbang/VideoCaptioning/829save/MSRVTT/zi/DFM_Model_grulstm_icas/A_SR8wt2_GFlow_RDirect_pEc_SS1_100_70__Avb256f260_EBN_ltm20_1002/",
    #"/home/yangbang/VideoCaptioning/829save/Youtube2Text/zi/DFM_Model_grulstm_ics/SR8_wMA0_wt0_SS1_pEc_100_70_EBN_1002/",
    #"/home/yangbang/VideoCaptioning/729save/Youtube2Text/zi/DFM_Model_grulstm_ics/SR8_wMA0_wt0_SS1_pEc_100_70_noHI_EBN_1002",
    #"/home/yangbang/VideoCaptioning/729save/Youtube2Text/zo/DFM_Model_grulstm_ics/SR8_wMA0_wt0_SS1_pEc_100_7_noHI_wise_EBN",
    
    #"/home/yangbang/VideoCaptioning/729save/Youtube2Text/zo/DFM_Model_grulstm_ics/SR8_wMA0_wt0_SS1_pEc_100_70_noHI_EBN_1037",
    #"/home/yangbang/VideoCaptioning/729save/Youtube2Text/zi/DFM_Model_grulstm_ics/SR8_wMA0_wt0_SS1_pEc_100_70_noHI_EBN_1037",


    #"/home/yangbang/VideoCaptioning/729save/MSRVTT/zo/DFM_Model_grulstm_ics/SR8_wMA0_wt2_SS1_pEc_100_70_noHI_EBN_ltm20_1002/",
    #"/home/yangbang/VideoCaptioning/729save/MSRVTT/zi/DFM_Model_grulstm_ics/SR8_wMA0_wt2_SS1_pEc_100_70_noHI_EBN_ltm20_1002/",

    #"/home/yangbang/VideoCaptioning/729save/MSRVTT/baseline/DFM_Model_linearlstm_icas/SR8_wMA0_wt2_SS1_pEc_together_100_70_noHI_wise_Avb256f260_EBN_ltm20/",


    #"/home/yangbang/VideoCaptioning/729save/MSRVTT/zo/DFM_Model_grulstm_icas/SR8_wMA0_wt2_SS1_pEc_100_70_noHI_wise_Avb256f260_EBN_ltm20/",
    #"/home/yangbang/VideoCaptioning/729save/MSRVTT/zi/DFM_Model_grulstm_icas/SR8_wMA0_wt2_SS1_pEc_100_70_noHI_wise_Avb256f260_EBN_ltm20/",


    #"/home/yangbang/VideoCaptioning/729save/MSRVTT/zo/DFM_Model_grulstm_icas/SR8_wMA0_wt2_SS1_pEc_100_70_noHI_wise_Avb256f260_EBN_ltm20/",
    #"/home/yangbang/VideoCaptioning/729save/MSRVTT/zi/DFM_Model_grulstm_icas/SR8_wMA0_wt2_SS1_pEc_100_70_noHI_Avb256f260_EBN_ltm20_1002/",


    ])
parser.add_argument('--ensemble_pretrained_paths', nargs='+', type=str, default=[
    '0_0377_176147_177127_173551_178530_178530.pth.tar',
    #'0_0461_255085_247368_231619_241061_236809.pth.tar', #0_0486_253423_250932_231401_239881_238204, 0_0461_255085_247368_231619_241061_236809
    #'0_0419_256027_253032_226829_239978_236815.pth.tar', #0_0390_255344_260957_229616_233767_234175, 0_0419_256027_253032_226829_239978_236815


    ])
parser.add_argument('--results_path', type=str, default='./ensemble_results')
parser.add_argument('--evaluate_mode', default='test', type=str)
parser.add_argument('--dataset', default='Youtube2Text', type=str)
parser.add_argument('--max_len', default=20, type=int)

parser.add_argument('--n_frames', default=-1, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--topk', default=1, type=int)
parser.add_argument('--beam_size', default=5, type=int)
parser.add_argument('--beam_candidate', default=5, type=int)
parser.add_argument('--beam_alpha', default=0.7, type=float)
parser.add_argument('--category_wise_eval', default=False, action='store_true')
parser.add_argument('--nf_eval', default=False, action='store_true')
parser.add_argument('--use_ltm', default=False, action='store_true')
parser.add_argument('--extra', default='', type=str)

args = parser.parse_args()
assert args.dataset in ['Youtube2Text', 'MSRVTT']
if args.category_wise_eval:
    assert args.dataset == 'MSRVTT'
args.modality = 'ic' if args.dataset == 'Youtube2Text' else 'ica'



if __name__ == "__main__":
    opt = vars(args)
    if len(opt['ensemble_pretrained_paths']) == 0:
        for i in range(len(opt['ensemble_checkpoint_paths'])):
            pth = os.path.join(opt['ensemble_checkpoint_paths'][i], 'best')
            fl = os.listdir(pth)
            best = 0
            best_fn = ''
            for filename in fl:
                if '.pth' in filename:
                    if int(filename.split('_')[5]) > best:
                        best = int(filename.split('_')[5])
                        best_fn = filename
            opt['ensemble_pretrained_paths'].append(best_fn)
    
    ensemble_evaluate(opt)
'''
if __name__ == "__main__":
    opt = vars(args)
    pth = os.path.join(opt['ensemble_checkpoint_paths'][0], 'best')
    tmp = os.listdir(pth)
    filelist1 = []
    for item in tmp:
        if '.pth' in item:
            filelist1.append(item)

    pth = os.path.join(opt['ensemble_checkpoint_paths'][1], 'best')
    tmp = os.listdir(pth)
    filelist2 = []
    for item in tmp:
        if '.pth' in item:
            filelist2.append(item)

    rec = []
    for i in range(len(filelist1)):
        for j in range(len(filelist2)):
            opt['ensemble_pretrained_paths'] = []
            opt['ensemble_pretrained_paths'].append(filelist1[i])
            opt['ensemble_pretrained_paths'].append(filelist2[j])
            res = ensemble_evaluate(opt)
            rec.append((i, j, res['Sum']))

    print(rec)
'''