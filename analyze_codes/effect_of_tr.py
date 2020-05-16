import numpy as np
import math
import pickle

sent1, score1 = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/all/MSRVTT_nv_AEmp_i5b6a135.pkl", 'rb'))
sent2, score2 = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/nt/MSRVTT_nv_AEmp_i5b6a135.pkl", 'rb'))

#print(sent1['video9999'][5::6])
#print(score1['video9999'][5::6])

index = 5
beam_alpha = 1.35
for key in sent1.keys():
    wtr = score1[key][index::6]
    ntr = score2[key][index::6]
    tmp = sent1[key][index::6]
    print('------------------ %s' % key)
    s1 = []
    s2 = []
    for ss1, ss2, t in zip(wtr, ntr, tmp):
        length = len(t.split(' '))
        s1.append(np.log(ss1).sum() / (length ** beam_alpha))
        s2.append(np.log(ss2).sum() / (length ** beam_alpha))

    s1 = -np.array(s1)
    s2 = -np.array(s2)
    #print(s1.tolist(), s2.tolist(), s1.shape)
    if np.argmin(s1) == np.argmin(s2):
        continue
    t1 = np.argsort(s1, kind='stable')
    t2 = np.argsort(s2, kind='stable')
    for i, sent in enumerate(sent1[key][index::6]):
        print("%90s %d %d" %(sent, t1.tolist().index(i)+1, t2.tolist().index(i)+1))
        #print(sent2[key][index::6][i])

    #for sent, idx1, idx2 in zip(sent1[key][index::6], np.argsort(s1, kind='stable'), np.argsort(s2, kind='stable')):
    #    print("%90s %d %d" %(sent, idx1+1, idx2+1))

print(sent1['video9999'])
print(sent1['video9999'][index::6])


index = 0
