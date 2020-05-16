import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

def cal_n_gram(data, n=4, with_key=True):
    gram_count = {}
    for key in data.keys():
        if with_key:
            cap = data[key][-1]['caption'].split(' ')
        else:
            cap = data[key][-1].split(' ')
        for i in range(len(cap) - n + 1):
            g = ' '.join(cap[i:i+n])
            gram_count[g] = gram_count.get(g, 0) + 1
    return gram_count
'''
ar_data = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/AR_topk_collect_results/msrvtt_1.pkl",'rb'))
gram_count = cal_n_gram(ar_data)
gram_count = sorted(gram_count.items(), key=lambda d: d[1], reverse=True)
sum1 = sum([item[1] for item in gram_count])
print(gram_count[:10], sum1, len(gram_count))

num_item = 100
data = [item[1]/sum1 for item in gram_count[:num_item]]
plt.bar(np.arange(num_item), data)
plt.show()
'''

def cal_n_gram_(data, n=4, with_key=True):
    gram_count = {}
    for key in data.keys():
        if with_key:
            cap = data[key][-1]['caption'].split(' ')
        else:
            cap = data[key][-1].split(' ')
        for i in range(len(cap) - n + 1):
            g = ' '.join(cap[i:i+n])
            gram_count[g] = gram_count.get(g, 0) + 1
    total_count = sum([item for k, item in gram_count.items()])
    gram_count = sorted(gram_count.items(), key=lambda d: d[1], reverse=True)
    gram_count = [item[1] for item in gram_count]
    res = [sum(gram_count[:item]) / total_count for item in range(len(gram_count))]

    return res


arb = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/AR_topk_collect_results/msrvtt_1.pkl",'rb'))
arb2 = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/AR_topk_collect_results/msrvtt_1_ag.pkl",'rb'))
cf_ctmp, _ = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/MSRVTT_nv_AEmp_i5b6a135.pkl", 'rb'))
#cf_ctef, _ = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/MSRVTT_nv_AEef_i1b6a135.pkl", 'rb'))

#cf_mp, _ = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/MSRVTT_nv_mp_i5b6a135.pkl", 'rb'))
#nab, _ = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/MSRVTT_mp_mp_i5b6a135.pkl", 'rb'))

data = [arb, arb2, cf_ctmp]
with_key = [True, True, False]
name = ['AR-B', 'AR-B2', 'NACF']
res = []
for item, wk in zip(data, with_key):
    res.append(cal_n_gram_(item, with_key=wk))

for item, n in zip(res, name):
    plt.plot([i for i in range(1, len(item)+1)], item, label=n)

plt.ylim(0, 1)
plt.xlim(0, 2500)
fig = plt.gcf()
fig.set_size_inches(3, 5)
plt.legend()
plt.show()