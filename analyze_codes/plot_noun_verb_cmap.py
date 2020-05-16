import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
import pickle
import nltk
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
    [["(", ")", "#"], "SS"]
]
for item in content:
    ks, v = item
    for k in ks:
        my_mapping[k] = v

refs = pickle.load(open("/home/yangbang/VideoCaptioning/MSRVTT/msrvtt_refs.pkl", 'rb'))

record_noun = {}
record_verb = {}
#interval = 0.1
limit = 7
count = 0
for key in tqdm(refs.keys()):
    #if int(key[5:]) > 1:
    #    continue
    for item in refs[key]:
        sent = item['caption'].split(' ')
        new_sent = []
        for w in sent:
            if w: new_sent.append(w)

        if len(new_sent) == limit:
            #print(new_sent)
            count+=1
            res = nltk.pos_tag(new_sent)

            for i, (w, t) in enumerate(res):
                #print(i, w, t)
                k = i+1
                if my_mapping[t] == 'NOUN':
                    record_noun[k] = record_noun.get(k, 0) + 1
                elif my_mapping[t] == 'VERB' and w not in ['is', 'are']:
                    #if k == 0:
                    #    print(w, k, i/len(res))
                    record_verb[k] = record_verb.get(k, 0) + 1

#noun = sorted(record_noun.items(), key=lambda d: d[0])
#verb = sorted(record_verb.items(), key=lambda d: d[0])
print(count)
noun, verb = record_noun, record_verb
sum1, sum2 = 0, 0

for i in range(1, limit+1):
    sum1 += noun.get(i, 0)
    sum2 += verb.get(i, 0)

for i in range(1, 1+limit):
    print("%2d\t%.3f\t%.3f" % (i, noun.get(i, 0) / sum1, verb.get(i, 0) / sum2))



'''


noun = [0.16383911551658406, 0.29713114754098363, 0.3310141059855128, 0.33373046130385053, 0.33473122378955394, 0.3342070148684712]
verb = [0.006385817765916889, 0.2049656881433473, 0.24904689287075868, 0.2719691193290126, 0.2788791460160122, 0.28054708349218455]
det = [0.3953488372093023, 0.3056138009912314, 0.28135722455203965, 0.2677277926038887, 0.26748951582157837]
ax = plt.subplot(1, 1, 1)
fontsize = 11


data = [noun, verb, det]
name = ['Noun', 'Verb', 'Determiner']

idx = [
    [i for i in range(len(noun))],
    [i for i in range(len(noun))],
    [i for i in range(1, len(noun))],
]

first = [
    1,
    1,
    0,
]

for i in range(len(data)):
    tmp = [item / data[i][first[i]] for item in data[i]]
    ax.plot(idx[i], tmp, label=name[i])

ax.xaxis.set_major_locator(x_major_locator)
#ax.yaxis.set_major_locator(y_major_locator) 
ax.set_xlim(0,5)
#ax.set_xticklabels(x_labels, rotation=60)
ax.set_ylabel('Relative Ratio', fontsize=fontsize)

fig = plt.gcf()
fig.set_size_inches(3.5, 2.5)
#fig.text(0.7, 0.16, 'determiner', fontsize=fontsize)
#fig.text(0.73, 0.33, 'noun(s)', fontsize=fontsize)

#fig.text(0.7, 0.66, 'preposition or', fontsize=fontsize)
#fig.text(0.62, 0.57, 'subordinating conjunction', fontsize=fontsize)

#fig.text(0.57, 0.85, 'gerund or present participle', fontsize=fontsize)


plt.subplots_adjust(left=0.20, bottom=0.21, right=0.98, top=0.97, wspace=0.35, hspace=None)

loc = (0.4, 0.2)
plt.annotate('', xy=(0, data[0][0]/data[0][1]), xytext=(0.4, 0.1), arrowprops=dict(arrowstyle='->', alpha=0.5))
plt.annotate('', xy=(0, data[1][0]/data[1][1]), xytext=(0.4, 0.1), arrowprops=dict(arrowstyle='->', alpha=0.5))
fig.text(0.26, 0.235, '  Attribute\nGeneration', fontsize=fontsize)
#fig.text(0.7, 0.5, 'Determiner', fontsize=fontsize)
#fig.text(0.72, 0.8, 'Noun', fontsize=fontsize)
#fig.text(0.4, 0.9, 'Verb', fontsize=fontsize)

plt.axhline(y=1, c="r", ls="--", lw=1)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('(b)', fontsize=fontsize+2)
plt.legend(loc='lower right')
plt.savefig('./b_pos_tagging.png')
plt.show()
'''