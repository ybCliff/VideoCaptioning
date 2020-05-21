import nltk
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

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
    [["UH"], "INTJ"],
    [['(', ')', '#'], "dummy"]
]
for item in content:
    ks, v = item
    for k in ks:
        my_mapping[k] = v

def check(data):
    rec = []
    for i in range(10):
        tmp = {}
        for item in content:
            ks, v = item
            tmp[v] = 0
        rec.append(tmp)


    total_len = 0
    nv_count = [0] * 10

    for k in data.keys():

        for i in range(len(data[k])):
            sent = data[k][i].split(' ')
            if i == 0:
                total_len += len(sent)

            res = nltk.pos_tag(sent)
            c = 0
            for w, t in res:
                if w != '<mask>':
                    rec[i][my_mapping[t]] += 1
                #if my_mapping[t] == 'ADJ':
                #    print(w)
                if i == 0:
                    if w != '<mask>':
                        c += 1
                else:
                    if my_mapping[t] in ['NOUN', 'VERB'] and w not in ['is', 'are', '<mask>']:
                        c += 1
            nv_count[i] += c

    print(total_len)
    noun = []
    verb = []
    det = []
    cconj = []
    for i in range(10):
        if nv_count[i]:
            print(i, nv_count[i]/total_len, nv_count[i])
            noun.append(rec[i]['NOUN'] / total_len)
            verb.append(rec[i]['VERB'] / total_len)
            det.append(rec[i]['DET'] / total_len)
            cconj.append((rec[i]['ADP']) / total_len)
    print(noun)
    print(verb)
    print(det)
    print(cconj)
    print('---------------------')

'''
NA2E = pickle.load(open("../iterative_collect_results/Youtube2Text_nv_AEmp_i5b5a100.pkl", 'rb'))[0]
NAB = pickle.load(open("../iterative_collect_results/Youtube2Text_mp_mp_i5b5a100.pkl", 'rb'))[0]
check(NA2E)
check(NAB)
'''
NA2E = pickle.load(open("../iterative_collect_results/MSRVTT_nv_AEmp_i5b6a135.pkl", 'rb'))[0]
#NAB = pickle.load(open("../iterative_collect_results/MSRVTT_mp_mp_i5b5a114.pkl", 'rb'))[0]
check(NA2E)
#check(NAB)


data1 = np.array([
        [0, 36.94],
        [1, 45.72],
        [2, 48.11],
        [3, 48.83],
        [4, 48.99],
        [5, 48.91]
    ])
data2 = np.array([
        [0, 17.02],
        [1, 36.22],
        [2, 44.28],
        [3, 46.81],
        [4, 47.62],
        [5, 47.73]
    ])
'''
index1 = np.arange(6)
index2 = np.arange(1, 6)
def change(data):
    new_data = [item * 100 for item in data]
    return np.array(new_data)

n1 = [0.275531914893617, 0.3021276595744681, 0.3071808510638298, 0.31090425531914895, 0.31196808510638296, 0.31063829787234043]
n1 = change(n1)
v1 = [0.09388297872340426, 0.3348404255319149, 0.35079787234042553, 0.35345744680851066, 0.3537234042553192, 0.35425531914893615]
v1 = change(v1)
d1 = [0.298936170212766, 0.2827127659574468, 0.2752659574468085, 0.27287234042553193, 0.27287234042553193]
d1 = change(d1)

n2 = [0.16383911551658406, 0.29713114754098363, 0.3310141059855128, 0.33373046130385053, 0.33473122378955394, 0.3342070148684712]
n2 = change(n2)
#[0.006385817765916889, 0.2049656881433473, 0.24904689287075868, 0.2719691193290126, 0.2788791460160122, 0.28054708349218455]
#[0.0, 0.3953488372093023, 0.3056138009912314, 0.28135722455203965, 0.2677277926038887, 0.26748951582157837]

fig,ax = plt.subplots(1, 2)
ax[0].plot(index1, n1, label='noun')
ax[0].plot(index1, v1, label='verb')
ax[0].plot(index2, d1, label='det')
ax[0].legend()

ax[1].plot(index1, n2)

#
#
'''


fig,ax = plt.subplots()

ax.plot(data1[:, 0], data1[:, 1], label='MSVD')
ax.plot(data2[:, 0], data2[:, 1], label='MSR-VTT')
ax.set_xlim([0,5]) 
ax.set_ylim([15,50])
ax.legend()
#ax.xaxis.set_minor_locator(MultipleLocator(0.25))
#ax.yaxis.set_minor_locator(MultipleLocator(0.05))

plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Percentage of nouns and verbs (%)", fontsize=14)
plt.tick_params(labelsize=12)
#plt.legend()
plt.subplots_adjust(top=0.97, bottom=0.11, right=0.97, left=0.13, hspace=0, wspace=0)
#fig = plt.gcf()
#fig.set_size_inches(6, 2)

#plt.margins(0, 0)
#plt.show()


data = pickle.load(open("/home/yangbang/VideoCaptioning/Youtube2Text/info_corpus_0.pkl", 'rb'))
info = data['info']
itow = info['itow']
caption = data['captions']
keylist = caption.keys()
rec = []

# for k in keylist:
#     for caps in caption[k]:
#         cap = [itow[item] for item in caps[1:-1]]
#         res = nltk.pos_tag(cap)
#         for w, t in res:
#             if my_mapping[t] == 'NOUN':
#                 rec.append(w)

record = {}
for k in tqdm(keylist):
    for caps in caption[k]:
        cap = [itow[item] for item in caps[1:-1]]
        res = nltk.pos_tag(cap)
        for w, t in res:
            if my_mapping[t] not in record.keys():
                record[my_mapping[t]] = []
            if w in ['<unk>']:
                continue
            if my_mapping[t] == 'VERB' and w in ['is', 'are', 'was', 'were']:
                continue
            record[my_mapping[t]].append(w)

r2 = {}
c = 0
for k, v in record.items():
    r2[k] = len(list(set(v))) #set(v)
    c += r2[k]

for k, v in r2.items():
    print(k, v/c)

print('---------------')

r2 = {}
c = 0
for k, v in record.items():
    r2[k] = len(list(v)) #set(v)
    c += r2[k]

for k, v in r2.items():
    print(k, v/c)




# for k, v in itow.items():
#     res = nltk.pos_tag([v])
#     for w, t in res:
#         if my_mapping[t] == 'NOUN':
#             rec.append(w)
# rec = list(set(rec))
# print(len(rec))

# with open('msvd_noun.txt', 'w') as f:
#     f.write('\n'.join(rec))


data = pickle.load(open("/home/yangbang/VideoCaptioning/MSRVTT/msrvtt_refs.pkl", 'rb'))
record = {}
print(type(data), len(data))
for i in tqdm(range(10000)):
    vid = 'video%d'%i
    for item in data[vid]:
        cap = item['caption'].split(' ')
        sent = [w for w in cap if w]
        res = nltk.pos_tag(sent)
        for w, t in res:
            if my_mapping[t] not in record.keys():
                record[my_mapping[t]] = []
            if my_mapping[t] == 'VERB' and w in ['is', 'are', 'was', 'were']:
                continue
            record[my_mapping[t]].append(w)



r2 = {}
c = 0
for k, v in record.items():
    r2[k] = len(list(set(v))) #set(v)
    c += r2[k]

for k, v in r2.items():
    print(k, v/c)