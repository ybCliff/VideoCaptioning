import json, os
import pickle
import numpy as np
def compare(d1, d2, t='CIDEr', keys=None):
  assert d1.keys() == d2.keys()
  c1, c2, c3 = 0, 0, 0
  if keys is None:
  	keys = d1.keys()
  
  total = len(keys)
  for k in keys:
    if d1[k][t] > d2[k][t]:
      c1 += 1
    elif d1[k][t] == d2[k][t]:
      c2 += 1
    else:
      c3 += 1
  print(c1/total, c2/total, c3/total)
    

d1 = json.load(open("/home/yangbang/VideoCaptioning/ARVC/nacf_ctmp_b6i5_135.json", 'r'))
d2 = json.load(open("/home/yangbang/VideoCaptioning/ARVC/nab_mp_b6i5_135.json", 'r'))
d3 = json.load(open("/home/yangbang/VideoCaptioning/ARVC/arb_b5.json", 'r'))
d4 = json.load(open("/home/yangbang/VideoCaptioning/ARVC/arb2_b5.json", 'r'))

metric = 'CIDEr'
compare(d1, d2, metric)
compare(d1, d3, metric)
compare(d1, d4, metric)


def compare2(d1, d2, d3, d4=None):
  c1, c2, c3 = 0, 0, 0
  total = len(d1.keys())
  t1 = 'CIDEr'
  t2 = 'Bleu_4'
  target = []
  for k in d1.keys():
    if d1[k][t1] > d2[k][t1] and d1[k][t1] > d3[k][t1]:
      if d1[k][t2] > d2[k][t2] and d1[k][t2] > d3[k][t2]:
        if d4 is not None:
          s = d4[k][0].split(' ')
          if s[2] != '<mask>' or s[3] != '<mask>':
            print(k)
            target.append(k)
        else:
          target.append(k)
          c1 += 1
  print(c1/total)
  return target

target = compare2(d1, d2, d3)
print(len(target))

remain = []
for k in d1.keys():
	if k not in target:
		remain.append(k)

#candidate = np.random.choice(target, 100).tolist()
#candidate += np.random.choice(remain, 400).tolist()

candidate = []
for i in range(60):
	while True:
		res = np.random.choice(remain, 1).tolist()[0]
		if (res+'.jpg') in os.listdir('qualitative_examples_target500'):
			continue
		else:
			break
	candidate.append(res)


#op = 'python generate_samples_to_compare.py --all --target %s'%(' '.join(candidate))
#os.system(op)



a, _ = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/MSRVTT_nv_AEmp_i5b6a135.pkl", 'rb'))
b, _ = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/MSRVTT_mp_mp_i5b6a135.pkl", 'rb'))
c = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/AR_topk_collect_results/msrvtt_1.pkl", 'rb'))
pth = 'qualitative_examples_target500'
repeated = 0
keys = []
#for f in os.listdir(pth):
#	key = f.split('.')[0]
for key in a.keys():
	keys.append(key)
	#if a[key][-1] == b[key][-1]:
	if a[key][-1] == c[key][0]['caption']:
		repeated += 1
print('repeated', repeated/len(os.listdir(pth)), repeated/len(a.keys()))
#compare(d1, d2, keys=keys)
#compare(d1, d3, keys=keys)


'''
mp, mp_s = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/MSRVTT_nv_AEmp_i5b6a135.pkl", 'rb'))
mpa, _ = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/all/MSRVTT_nv_mp_i5b6a135.pkl", 'rb'))


target = compare2(d1, d2, d3)
#target = compare2(d1, d2, d3, mp)



#op = 'python generate_samples_to_compare.py --all --target %s'%(' '.join(target))
#os.system(op)

def find_(data, tl):
  sub = [item for item in data if len(item.split(' ')) == tl]
  #for item in sub:
  #  print(item)
  return sub[-1]


keylist = ['video7206','video7881','video9843','video9171','video9396','video7339']
#keylist = mp.keys()
c = 0
for k in target:#keylist:
  length = len(mp[k][-1].split(' '))
  s = find_(mpa[k], length)
  if mp[k][-1] != s and 'talk show' in mp[k][-1]:
    print(k, length, mp[k][-1], s)
  c+=1
print(c)


mp, mp_s = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/Youtube2Text_nv_AEmp_i5b5a100.pkl", 'rb'))
mpa, _ = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/all/Youtube2Text_nv_mp_i5b5a100.pkl", 'rb'))
nab, _ = pickle.load(open("/home/yangbang/VideoCaptioning/ARVC/iterative_collect_results/Youtube2Text_mp_mp_i5b5a100.pkl", 'rb'))
keylist = ['video1311','video1808','video1377']

for k in keylist:
  length = len(mp[k][-1].split(' '))
  s = find_(mpa[k], length)
  print(k, length, mp[k][-1], s, nab[k][-1])
'''