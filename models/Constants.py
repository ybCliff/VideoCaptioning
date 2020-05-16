
PAD = 0
UNK = 1
BOS = 2
EOS = 3
MASK = 4

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
MASK_WORD = '<mask>'

'''
PAD = 12594
BOS = 12595
EOS = 0
  
PAD_WORD = '<pad>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
'''

mapping = {
    'lang': ('seq_probs', 'gold'),
    'obj': ('pred_obj', 'obj'),
    'tag': ('pred_tags', 'taggings'),
    'length': ('pred_length', 'length_target'),
    'bow': ('scores', 'target'),
    'attr': ('pred_attr', 'attribute'),
    'dist': ('pred_embs', 'bert_embs'),
    'attr2': ('pred_attr2', 'attribute'),
}
'''
EOS = 0
BOS = 1

BOS_WORD = '<sos>'
EOS_WORD = '<eos>'
'''