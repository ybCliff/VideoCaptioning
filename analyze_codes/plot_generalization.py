import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
x_major_locator=MultipleLocator(1)

x = np.arange(3)
x_labels = ['', 'AR-B', 'NA-B', 'NA2E']
bar_width = 0.3
rotation = 0
fontsize = 11
#beam_alpha 1.0 1.15, 1.0
'''
bleu_msvd_msvd = [48.7, 51.4, 52.4]
bleu_vtt_msvd = [34.54, 37.44, 37.91]

ax = plt.subplot(111)
ax.bar(x-bar_width/2, bleu_msvd_msvd, bar_width, label='Train: MSVD, Test: MSVD')
ax.bar(x+bar_width/2, bleu_vtt_msvd, bar_width,  label='Train: MSR-VTT, Test: MSVD')
for a, b in zip(x, bleu_msvd_msvd):
    ax.text(a-0.3, b+0.12, "%.1f"%b, fontsize=fontsize)
for i, (a, b) in enumerate(zip(x, bleu_vtt_msvd)):
    if i == 3:
        ax.text(a, b+0.12, "%.1f"%b, fontsize=fontsize, color='r', fontweight='bold')
    else:
        ax.text(a, b+0.12, "%.1f"%b, fontsize=fontsize)

ax.set_ylim(33, 55)
ax.set_xticklabels(x_labels, rotation=rotation, fontsize=fontsize)
ax.xaxis.set_major_locator(x_major_locator)
ax.set_ylabel('BLEU@4', fontsize=fontsize)
#ax.legend()
ax.legend(loc='center right')
plt.subplots_adjust(left=0.15, bottom=0.2, right=0.99, top=0.97, wspace=0.25, hspace=None)
plt.xlabel("(a)", fontsize=fontsize+2)
plt.yticks(fontsize=fontsize)
fig = plt.gcf()
fig.set_size_inches(3.5, 2.5)
plt.savefig('./a_generalization_bleu.png')
plt.show()

'''
cider_msvd_msvd = [91.8, 92.5, 96.1]
cider_vtt_msvd = [55.17, 56.19, 58.94]
ax = plt.subplot(111)
ax.bar(x-bar_width/2, cider_msvd_msvd, bar_width, label='Train: MSVD, Test: MSVD')
ax.bar(x+bar_width/2, cider_vtt_msvd, bar_width,  label='Train: MSR-VTT, Test: MSVD')
for i, (a, b) in enumerate(zip(x, cider_msvd_msvd)):
    if i == 2:
        ax.text(a, b-0.8, "%.1f"%b, fontsize=fontsize)
    else:
        ax.text(a-0.3, b+0.2, "%.1f"%b, fontsize=fontsize)
for i, (a, b) in enumerate(zip(x, cider_vtt_msvd)):
    if i == 3:
        ax.text(a, b+0.12, "%.1f"%b, fontsize=fontsize, color='r', fontweight='bold')
    else:
        ax.text(a, b+0.12, "%.1f"%b, fontsize=fontsize)

ax.set_ylim(54, 99)
ax.set_xticklabels(x_labels, rotation=rotation, fontsize=fontsize)
ax.xaxis.set_major_locator(x_major_locator)
ax.set_ylabel('CIDEr-D', fontsize=fontsize)
#ax.legend()
ax.legend(loc='center right')
plt.subplots_adjust(left=0.15, bottom=0.2, right=0.99, top=0.97, wspace=0.25, hspace=None)
plt.xlabel("(b)", fontsize=fontsize+2)
plt.yticks(fontsize=fontsize)
fig = plt.gcf()
fig.set_size_inches(3.5, 2.5)
plt.savefig('./b_generalization_cider.png')
plt.show()
