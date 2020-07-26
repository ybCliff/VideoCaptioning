import torch
import torch.nn as nn
import numpy as np
import random
import os
import models.Constants as Constants


def set_seed(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    if not isinstance(seq, np.ndarray):
        seq = seq.cpu()
    N, D = seq.shape[0], seq.shape[1]
   #D = 28
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].item()
            if ix != Constants.EOS:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[ix]
            else:
                break
        out.append(txt)
    return out

def get_words_with_specified_tags(word_to_ix, seq, index_set, demand=['NOUN', 'VERB'], ignore_words=['is', 'are', '<mask>']):
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
    assert isinstance(index_set, set)
    res = nltk.pos_tag(seq.split(' '))
    for w, t in res:
        if my_mapping[t] in demand and w not in ignore_words:
            index_set.add(word_to_ix[w])



def main():
    import numpy as np
    import torch.nn.functional as F

    a = np.array([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]])
    logpt = F.log_softmax(torch.from_numpy(a), dim=2)
    print(logpt)
    target = torch.Tensor([[3, 4, 5, 3]]).type(torch.LongTensor)
    mask = torch.Tensor([[1, 1, 1, 0]]).type(torch.DoubleTensor)
    print(a.shape, target.shape, mask.shape)

    myLoss = LanguageModelCriterion(use_focalloss=False)
    print(myLoss(logpt, target, mask))
    

if __name__ == '__main__':
    main()