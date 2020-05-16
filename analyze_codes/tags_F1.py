import nltk
import pickle
from collections import defaultdict
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
    [["(", ")", "#"], "dummy"]
]
for item in content:
    ks, v = item
    for k in ks:
        my_mapping[k] = v

def calculate_precision_and_recall(gts, hyp, freq):
	correct = 0
	count = 0
	count_gt = 0

	correct_r = 0
	count_r = 0
	for k in hyp.keys():
		tmp = 0
		count += len(hyp[k])
		count_gt += len(gts[k])
		for item in gts[k]:
			count_r += freq[k][item]
		for tag in hyp[k]:
			if tag in gts[k]:
				correct += 1
				correct_r += freq[k][tag]
				tmp += 1
		#if len(hyp[k]) > 10 and tmp / len(hyp[k]) > 0.8:
		#	print(k)
	#print(correct, count, correct/count)
	#print(correct_r, count_r, correct_r/count_r)

	return correct/count, correct/count_gt
	#return correct/count, correct_r/count_r

def combine(data, keylist):
	word_set = set()
	for k in keylist:
		v = data[k]
		for item in v:
			word_set.add(item)
	return {'dummy': list(word_set)}

def calculate_F1(gts, hyp, freq):
	precision, recall = calculate_precision_and_recall(gts, hyp, freq)
	f1 = 2 * precision * recall / (precision + recall)
	print('F1: %.2f\tPrecision: %.2f\tRecall: %.2f' % (100 * f1, 100 * precision, 100 * recall))

'''
def load_tags(data, keylist, with_key=False, frequency=False):
	tags = {}
	freq = {}
	count = 0
	for k in keylist:
		v = data[k]
		word_set = set()
		for s in v:
			if with_key:
				s = s['caption']
			#print(s, s.split(' '))
			s = s.split(' ')
			new_s = []
			for item in s:
				if item:
					new_s.append(item)
			res = nltk.pos_tag(new_s)
			
			for w, t in res:
				if my_mapping[t] in ['NOUN', 'VERB'] and w not in ['are', 'is', '<unk>', '<mask>']:
					if frequency:
						count += 1
						freq[w] = freq.get(w, 0) + 1
					word_set.add(w)
		tags[k] = list(word_set)
	if frequency:
		for k in freq.keys():
			freq[k] = count / freq[k]
		return tags, freq
	return tags
'''
def load_tags(data, keylist, with_key=False, frequency=False):
	tags = {}
	freq = {}
	count = {}
	for k in keylist:
		v = data[k]
		word_set = set()
		for s in v:
			if with_key:
				s = s['caption']
			#print(s, s.split(' '))
			s = s.split(' ')
			new_s = []
			for item in s:
				if item:
					new_s.append(item)
			res = nltk.pos_tag(new_s)
			
			for w, t in res:
				if my_mapping[t] in ['NOUN', 'VERB'] and w not in ['are', 'is', '<unk>', '<mask>']:
					if frequency:
						count[k] = count.get(k, 0) + 1
						if k not in freq.keys(): freq[k] = {}
						freq[k][w] = freq[k].get(w, 0) + 1
					word_set.add(w)
		tags[k] = list(word_set)
	if frequency:
		for k in freq.keys():
			for w in freq[k].keys():
				freq[k][w] /= count[k]
		return tags, freq
	return tags

def element_wise_F1(gts, hyp):
	f1 = {}
	for k in hyp.keys():
		correct = 0
		count = len(hyp[k])
		count_gt = len(gts[k])

		for tag in hyp[k]:
			if tag in gts[k]:
				correct += 1

		p = correct/count
		r = correct/count_gt
		f1[k] = 2 * p * r / (p + r + 0.0000001)
	return f1

if __name__ == '__main__':

	'''
	gts = pickle.load(open("/home/yangbang/VideoCaptioning/Youtube2Text/msvd_refs.pkl", 'rb'))
	hyp = pickle.load(open('../iterative_collect_results/all/Youtube2Text_nv_AEmp_i5b5a100.pkl', 'rb'))[0]
	hyp2 = pickle.load(open('../iterative_collect_results/all/Youtube2Text_mp_mp_i5b5a100.pkl', 'rb'))[0]
	hyp3 = pickle.load(open('../AR_topk_collect_results/msvd_5.pkl', 'rb'))
	'''
	gts = pickle.load(open("/home/yangbang/VideoCaptioning/MSRVTT/msrvtt_refs.pkl", 'rb'))
	hyp = pickle.load(open('../iterative_collect_results/all/MSRVTT_nv_AEmp_i5b5a114.pkl', 'rb'))[0]
	hyp2 = pickle.load(open('../iterative_collect_results/all/MSRVTT_mp_mp_i5b5a114.pkl', 'rb'))[0]
	hyp3 = pickle.load(open('../AR_topk_collect_results/msrvtt_5.pkl', 'rb'))



	#tags_hyp = load_tags(hyp, hyp.keys(), with_key=False)
	tags_hyp2 = load_tags(hyp2, hyp.keys(), with_key=False)
	#tags_hyp3 = load_tags(hyp3, hyp.keys(), with_key=True)
	#tags_gts, freq = load_tags(gts, hyp.keys(), with_key=True, frequency=True)
	print(tags_hyp2['video8335'])
	'''
	calculate_F1(tags_gts, tags_hyp, freq)
	calculate_F1(tags_gts, tags_hyp2, freq)
	calculate_F1(tags_gts, tags_hyp3, freq)
	'''

	f1 = element_wise_F1(tags_gts, tags_hyp)
	f2 = element_wise_F1(tags_gts, tags_hyp3)
	#res = sorted(f1.items(), key=lambda d: d[1])
	'''
	for item in res[::-1][:200]:
		print(item)
		print(tags_hyp[item[0]])
		print(tags_hyp3[item[0]])
	'''

	res = {}
	for k in f1.keys():
		res[k] = f1[k] - f2[k]
	res = sorted(res.items(), key=lambda d: d[1])[::-1]

	for item in res[:200]:
		print(item)
		print(tags_hyp[item[0]])
		print(tags_hyp3[item[0]])
	#print(res[:10])