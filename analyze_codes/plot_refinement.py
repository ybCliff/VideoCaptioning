import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

x_major_locator=MultipleLocator(1)
y_major_locator=MultipleLocator(0.05)
'''
ax = plt.subplot(1, 1, 1)
fontsize = 11

#B4 = [33.38, 38.48, 41.23, 41.73, 42.55, 42.29, 42.06, 42.48, 42.57, 42.39]
#M = [25.18, 26.83, 27.97, 28.37, 28.69, 28.76, 28.70, 28.65, 28.61, 28.67]
#R = [60.13, 60.84, 61.71, 61.95, 62.31, 62.16, 61.92, 61.81, 61.84, 61.79]
#C = [43.71, 47.87, 49.76, 50.81, 51.46, 51.39, 51.12, 51.35, 50.87, 50.92]

#B4 = [33.38, 38.48, 41.23, 41.73, 42.55, 42.29]
#M = [25.18, 26.83, 27.97, 28.37, 28.69, 28.76]
#R = [60.13, 60.84, 61.71, 61.95, 62.31, 62.16]
#C = [43.71, 47.87, 49.76, 50.81, 51.46, 51.39]


data = [B4, C, M, R]
name = ['BLEU@4', 'CIDEr-D', 'METEOR', 'ROUGE-L']

idx = [i + 1 for i in range(len(M))]

for i in range(len(data)):
    tmp = [item / data[i][0] for item in data[i]]
    ax.plot(idx, tmp, label=name[i])

ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_minor_locator(y_major_locator) 
ax.set_xlim(1,len(B4))
#ax.set_xticklabels(x_labels, rotation=60)
ax.set_ylabel('Relative Improvement', fontsize=fontsize)
ax.set_ylim(1.0, 1.6)
fig = plt.gcf()
fig.set_size_inches(3.5, 2.5)
#fig.text(0.7, 0.16, 'determiner', fontsize=fontsize)
#fig.text(0.73, 0.33, 'noun(s)', fontsize=fontsize)

#fig.text(0.7, 0.66, 'preposition or', fontsize=fontsize)
#fig.text(0.62, 0.57, 'subordinating conjunction', fontsize=fontsize)

#fig.text(0.57, 0.85, 'gerund or present participle', fontsize=fontsize)


plt.subplots_adjust(left=0.20, bottom=0.21, right=0.98, top=0.97, wspace=0.35, hspace=None)
#plt.axhline(y=1, c="r", ls="--", lw=1)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(loc='upper left')
plt.xlabel('(a)', fontsize=fontsize+2)
#plt.legend(loc='lower right')
plt.savefig('./a_iterative_refinement.png')
plt.show()
'''


noun = [0.1589141872701029, 0.2904036876658751, 0.326954416352377, 0.33030683987521536, 0.33151743725846256, 0.33151743725846256]
verb = [0.007123899986031568, 0.19946919960888393, 0.2484052707547609, 0.2692647948968664, 0.2752712203752852, 0.2765283791963496]
det = [0.4078781952786702, 0.30693299809098107, 0.28071890859989757, 0.26735577594636123, 0.2672160916329096]

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
ax.set_ylabel('Relative Change Rate', fontsize=fontsize)

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
fig.text(0.26, 0.235, 'Visual Word\nGeneration', fontsize=fontsize)
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

b4 = [0.35102615515663516, 0.39044594766030033, 0.4044098749369502, 0.4123103640799696, 0.4187011223943814, 0.42032901389131955, 0.4200607284644363, 0.4231304753146409, 0.4203992606520593, 0.4179614896563713]
m = [0.2618245886638457, 0.27910513813112753, 0.28374279808723046, 0.28610763456706934, 0.2867498017446285, 0.28731346803183283, 0.28655384452010585, 0.28335965638915117, 0.280912590293081, 0.28021448767497087]
r = [0.5970163114092318, 0.6118677188426259, 0.6179437174009608, 0.6194298949490724, 0.6212017346607663, 0.6220326428034066, 0.6237129665066987, 0.6219345543238958, 0.6203223169434933, 0.6198689725634154]
c = [0.4350628978487853, 0.4769514163268435, 0.49666838026139504, 0.5050747491302513, 0.5106524374147738, 0.5142706105602625, 0.5158804356222461, 0.5105365581733672, 0.5061391444554943, 0.5054068744214035]
l = [6.7157190635451505, 7.2792642140468224, 7.2612040133779265, 7.282274247491639, 7.233779264214047, 7.182943143812709, 7.030100334448161, 6.557190635451505, 6.211705685618729, 6.097993311036789]

ax = plt.subplot(1, 1, 1)
fontsize = 11


data = [b4, m, r, c, l]
name = ['BLEU@4', 'METEOR', 'ROUGE-L', 'CIDEr-D', 'avg. length']

idx = [
    [i for i in range(len(b4))] for _ in range(5)
]

first = [
    0,
    0,
    0,
    0,
    0
]

for i in range(len(data)):
    tmp = [item / data[i][first[i]] for item in data[i]]
    ax.plot(idx[i], tmp, label=name[i])

ax.xaxis.set_major_locator(x_major_locator)
#ax.yaxis.set_major_locator(y_major_locator) 
ax.set_xlim(0,9)
#ax.set_xticklabels(x_labels, rotation=60)
ax.set_ylabel('Relative Ratio', fontsize=fontsize)

fig = plt.gcf()
fig.set_size_inches(7, 2)
#fig.text(0.7, 0.16, 'determiner', fontsize=fontsize)
#fig.text(0.73, 0.33, 'noun(s)', fontsize=fontsize)

#fig.text(0.7, 0.66, 'preposition or', fontsize=fontsize)
#fig.text(0.62, 0.57, 'subordinating conjunction', fontsize=fontsize)

#fig.text(0.57, 0.85, 'gerund or present participle', fontsize=fontsize)


plt.subplots_adjust(left=0.20, bottom=0.21, right=0.98, top=0.97, wspace=0.35, hspace=None)

plt.axhline(y=1, c="r", ls="--", lw=1)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('(b)', fontsize=fontsize+2)
plt.legend(loc='lower right')
#plt.savefig('./b_pos_tagging.png')
plt.show()
'''