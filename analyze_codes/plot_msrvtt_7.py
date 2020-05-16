import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
from matplotlib.ticker import MultipleLocator
import itertools


data = pickle.load(open("/home/yangbang/VideoCaptioning/MSRVTT/info_corpus_2.pkl", 'rb'))
#split = data['info']['split']['train']
split = [i for i in range(10000)]
caps = data['captions']


vocab_size = 10546
target_length = 7
rec = [[] for _ in range(target_length)]
count = 0

for i in split:
  vid = 'video%d'%i
  cap = caps[vid]
  for item in cap:
    c = item[1:-1]
    if len(c) == target_length:
      count += 1
      for j in range(target_length):
        rec[j].append(c[j])

res = []
for i in range(target_length):
  tmp = list(set(rec[i]))
  res.append(len(tmp) / vocab_size)

print(res)
print(count)


