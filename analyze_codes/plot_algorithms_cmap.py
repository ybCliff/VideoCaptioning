import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
from matplotlib.ticker import MultipleLocator
import itertools

def plot_confusion_matrix(cm, x, y,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          fontsize=12,
                          subplot=111
                          ):
    """
    绘制混淆矩阵
        - cm: 混淆矩阵，shape like [N, N]
        - classses: 类别列表, len(classes) = N
        - normalize: True(归一化)，False(绝对值)
        - title: 图的标题名字
        - cmap: 使用哪种类型的渐变色
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #cm = cm.astype('float') / cm.max(axis=1)[:, np.newaxis]
    cm = cm.transpose()

    plt.subplot(subplot)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=fontsize)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=fontsize-2)
    #plt.colorbar(fraction=0.046, pad=0.02)

    plt.xticks(np.arange(len(x)), x, fontsize=fontsize)
    plt.yticks(np.arange(len(y)), y, fontsize=fontsize)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", 
                 fontsize=fontsize-2)

    #plt.tight_layout()
    plt.xlabel('$t$-th iteration', fontsize=fontsize)
    plt.ylabel('$n$-th position', fontsize=fontsize)
    

def run(data, score, target_length, num):
    count = {}
    rec = [[0 for i in range(target_length)] for _ in range(num)]
    print(len(rec), len(rec[0]))

    for key in data.keys():
        cap = data[key][-1]
        length = len(cap.split(' '))
        count[length] = count.get(length, 0) + 1
        if length != target_length:
            continue

        index = -1
        s = score[key]
        for i in range(1, len(data[key])):
            if s[i] != s[i-1]:
                index += 1
            for j in range(length):
                if s[i][j] != s[i-1][j]:
                    rec[index][j] += 1

    return count, np.array(rec)

target_length = 7
mp, mp_s = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/MSRVTT_nv_AEmp_i5b6a135.pkl", 'rb'))
#ef, ef_s = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/MSRVTT_nv_AEef_i1b6a135.pkl", 'rb'))
ef, ef_s = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/MSRVTT_nv_ef_i0b6a135.pkl", 'rb'))
nab, nab_s = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/MSRVTT_nv_mp_i5b6a135.pkl", 'rb'))

c1, r1 = run(mp, mp_s, target_length, 5)
c2, r2 = run(ef, ef_s, target_length, 7)
c3, r3 = run(nab, nab_s, target_length, 4)

for i in range(10):
    print(i, c1.get(i, 0), c2.get(i, 0), c3.get(i, 0))

print(r1)
print(r2)
print(r3)

fontsize = 15
'''
plot_confusion_matrix(r2, ['%d'%i for i in range(1, target_length+1)], ['%d'%i for i in range(1, target_length+1)],
                          normalize=True,
                          title='(a) EF algorithm ($q=1$)',
                          cmap=plt.cm.Blues, subplot=111, fontsize=fontsize)

plot_confusion_matrix(r1, ['%d'%i for i in range(1, 6)], ['%d'%i for i in range(1, target_length+1)],
                          normalize=True,
                          title='(b) CT-MP algorithm ($T=5$)',
                          cmap=plt.cm.Blues, subplot=111,fontsize=fontsize)
'''

plot_confusion_matrix(r3, ['%d'%i for i in range(1, 5)], ['%d'%i for i in range(1, target_length+1)],
                          normalize=True,
                          title='(b) MP algorithm ($T=5$)',
                          cmap=plt.cm.Blues, subplot=111,fontsize=fontsize)


fig = plt.gcf()
fig.set_size_inches(5.5, 5)
plt.subplots_adjust(top=0.95, bottom=0.12, right=0.99, left=0.1, hspace=0, wspace=0)
#fig.savefig('./a_ef_cmap.png')
#plt.subplots_adjust(top=0.95, bottom=0.12, right=0.99, left=0.08, hspace=0, wspace=0)
#fig.savefig('./b_ctmp_cmap.png')
plt.show()