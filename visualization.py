import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default="/home/yangbang/VideoCaptioning/529save/MSRVTT/")
parser.add_argument('--scope', nargs='+', type=str, default=['S2ADRM_boaw_vggish2branchAdj_BN', 'S2ADRM_boaw_vggish2branchAdj'])
parser.add_argument('--filename', type=str, default='trainning_record.csv')
parser.add_argument('--legend', nargs='+', type=str, default=['w/ BN', 'w/o BN'])
parser.add_argument('--context', nargs='+', type=str, default=['train_loss'], help='train_loss | val_loss Bleu_4 | METEOR | ROUGE_L | CIDEr | Sum')
#parser.add_argument('--info', nargs='+', type=str, default=['Performance on MSR-VTT Validation Set', '', 'epoch'], help='title | ylabel | xlabel')
parser.add_argument('--info', nargs='+', type=str, default=['Training Loss Curve on MSR-VTT', '', 'epoch'], help='title | ylabel | xlabel')
parser.add_argument('--index', type=str, default='epoch')
parser.add_argument('--scope_info', nargs='+', type=str, default=['DFM_Model_gru', 'ica', 'ar', '30'])

args = parser.parse_args()

def main():
    csv = []
    for i in range(len(args.scope)):
        scope = os.path.join(args.scope_info[0], args.scope_info[1], args.scope_info[2], args.scope[i], args.scope_info[3])
        csv.append(pd.read_csv(os.path.join(args.base_path, scope, args.filename)))
    
    #train_loss val_loss Bleu_4  METEOR  ROUGE_L CIDEr Sum
    # title ylabel xlabel

    data = {}
    for i in range(len(csv)):
        for j in range(len(args.context)):
            final_legend = args.legend[i] + '_' + args.context[j]
            data[final_legend] = csv[i][args.context[j]]

    for i in range(len(csv) - 1):
        assert csv[i][args.index].equals(csv[i+1][args.index])

    data[args.index] = csv[0][args.index]

    data = pd.DataFrame(data)
    data = data.set_index(args.index)
    print(data.index)
    data.plot()

    plt.title(args.info[0])
    plt.ylabel(args.info[1])
    plt.xlabel(args.info[2])

    plt.show() 

if __name__ == '__main__':
    assert len(args.info) == 3
    assert len(args.scope) == len(args.legend)
    main()

'''
python visualization.py \
--scope ICDkdoubleBN_l_tml25_gg_ar ICDkdoubleBN09_l_tml25_gg_ar ICDkdoubleBN01_l_tml25_gg_ar \
--legend 2BN05 2BN09 2BN01 \
--context Bleu_4 METEOR ROUGE_L CIDEr \
--info Metric_Curve \'\' epoch \
'''