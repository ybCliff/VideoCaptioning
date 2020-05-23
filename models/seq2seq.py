import torch
import torch.nn as nn
import torch.nn.functional as F
import models.Constants as Constants
from tqdm import tqdm
import math
from torch.nn.parameter import Parameter

class Seq2Seq(nn.Module):
    def __init__(self, 
                preEncoder=None, 
                encoder=None, 
                joint_representation_learner=None, 
                auxiliary_task_predictor=None, 
                decoder=None, 
                tgt_word_prj=None, 
                beam_decoder=None,
                opt={}
            ):
        super(Seq2Seq, self).__init__()
        self.preEncoder = preEncoder
        self.encoder = encoder
        self.joint_representation_learner = joint_representation_learner
        self.auxiliary_task_predictor = auxiliary_task_predictor
        self.decoder = decoder
        #if opt.get('others', False):
        self.tgt_word_prj = getattr(self.decoder.embedding, 'linear', None)
        if self.tgt_word_prj is None:
            self.tgt_word_prj = nn.Linear(opt['dim_word'], opt['vocab_size'], bias=False)
            nn.init.xavier_normal_(self.tgt_word_prj.weight)
        #else:
        #    self.tgt_word_prj = tgt_word_prj
        #    nn.init.xavier_normal_(self.tgt_word_prj.weight)

        self.encoder_type = opt['encoder_type']
        self.decoder_type = opt['decoder_type']
        self.addition = opt.get('addition', False)
        self.temporal_concat = opt.get('temporal_concat', False)
        self.opt = opt
        '''
        if opt['use_tag']:
            self.tag2hidden = nn.Linear(opt['dim_t'], opt['dim_hidden'])
            nn.init.xavier_normal_(self.tag2hidden.weight)
        else:
            self.tag2hidden = None
        '''
        self.tag2hidden = None

        if opt.get('knowledge_distillation_with_bert', False):
            self.bert_embs_to_hidden = nn.Linear(opt['dim_bert_embeddings'], opt['dim_hidden'])
            nn.init.xavier_normal_(self.bert_embs_to_hidden.weight)
            self.bert_dropout = nn.Dropout(0.3)
            self.embs_dropout = nn.Dropout(0.3)
        else:
            self.bert_embs_to_hidden = None

        if opt.get('multitask_attribute', False):
            self.attribute_predictor = nn.Sequential(
                    nn.Linear(opt['dim_hidden'], opt.get('dim_t', 1000), bias=False),
                    nn.Sigmoid()
                )
        else:
            self.attribute_predictor = None
        self.beam_decoder = beam_decoder


    def encode(self, feats, semantics=None):
        results = {}

        if self.preEncoder is not None:
            feats = self.preEncoder(input_feats=feats, semantics=semantics)

        gate = None
        #if self.encoder.will_return_gate_info():
        #    encoder_outputs, encoder_hiddens, gate = self.encoder(feats)#, semantics=semantics)
        #else:
        encoder_outputs, encoder_hiddens = self.encoder(feats)#, semantics=semantics)
        

        if self.joint_representation_learner is not None:
            encoder_outputs, encoder_hiddens = self.joint_representation_learner(encoder_outputs, encoder_hiddens)

        if self.auxiliary_task_predictor is not None:
            auxiliary_results = self.auxiliary_task_predictor(encoder_outputs)
            results.update(auxiliary_results)

            if self.tag2hidden is not None:
                pred_attr = auxiliary_results.get('pred_attr', None)
                assert pred_attr is not None
                encoder_outputs += self.tag2hidden(pred_attr).unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)

        results['enc_output'] = encoder_outputs
        results['enc_hidden'] = encoder_hiddens
        results['gate'] = gate
            
        return results

    def forward(self, **kwargs):
        if self.opt.get('use_rl', False):
            func_mapping = {
                'ARFormer': self.forward_ARFormer_rl,
                'LSTM': self.forward_LSTM_rl,
            }
        else:
            func_mapping = {
                'LSTM': self.forward_LSTM,
                'ARFormer': self.forward_ARFormer,
                'NARFormer': self.forward_NARFormer,
                'ENSEMBLE': self.forward_ENSEMBLE
            }

        if self.opt.get('use_beam_decoder', False):
            return self.forward_beam_decoder(kwargs)
        
        return func_mapping[kwargs['opt']['decoder_type']](kwargs)

    
    
    def forward_NARFormer(self, kwargs):
        feats, tgt_tokens, category, vocab, tags, bert_embs = map(
            lambda x: kwargs[x], 
            ["feats", "tgt_tokens", "category", "vocab", "tags", "bert_embs"]
        )

        encoder_outputs = self.encode(feats)
        if self.opt['method'] == 'direct' or self.opt['method'] == 'signal' or self.opt['method'] == 'signal3' or self.opt['method'] == 'signal2' or self.opt['method'] == 'nv' or self.opt['method'] == 'ms':
            seq_probs, pred_embs, *_ = self.decoder(tgt_tokens, encoder_outputs['enc_output'], category, self.tgt_word_prj, vocab=vocab, tags=tags)#tags=encoder_outputs.get(Constants.mapping['attr'][0], None))
        else:
            seq_probs, pred_embs, *_ = self.decoder(tgt_tokens, encoder_outputs['enc_output'], category, tags=tags)#, tags=encoder_outputs.get(Constants.mapping['attr'][0], None))#, self.tgt_word_prj, vocab=vocab)

        scores = self.tgt_word_prj(seq_probs[0]) if isinstance(seq_probs, list) else self.tgt_word_prj(seq_probs)
        #pred_embs = seq_probs[-1] if isinstance(seq_probs, list) else seq_probs
        #pred_embs = self.pred_embs_to_hidden(pred_embs)

        if isinstance(seq_probs, list):
            res = []
            for i in range(len(seq_probs)):
                res.append(F.log_softmax(self.tgt_word_prj(seq_probs[i]), dim=-1))
            seq_probs = res
        else:
            seq_probs = F.log_softmax(self.tgt_word_prj(seq_probs), dim=-1)

        assert encoder_outputs['pred_length'] is not None

        if bert_embs is not None:
            assert self.bert_embs_to_hidden is not None
            
            bert_embs = self.bert_embs_to_hidden(bert_embs)
            bert_embs = self.bert_dropout(bert_embs)
            pred_embs = self.embs_dropout(pred_embs)

        if self.attribute_predictor is not None:
            pred_attr = self.attribute_predictor(pred_embs)
        else:
            pred_attr = None

        return {Constants.mapping['lang'][0]: seq_probs, Constants.mapping['length'][0]: encoder_outputs['pred_length'], 
                Constants.mapping['bow'][0]: scores, Constants.mapping['attr'][0]: encoder_outputs.get(Constants.mapping['attr'][0], None),
                Constants.mapping['dist'][0]: pred_embs, Constants.mapping['dist'][1]: bert_embs,
                Constants.mapping['attr2'][0]: pred_attr
                }

        

    def forward_ARFormer(self, kwargs):
        feats, tgt_tokens, category, bert_embs = map(
            lambda x: kwargs[x], 
            ["feats", "tgt_tokens", "category", "bert_embs"]
        )

        encoder_outputs = self.encode(feats)

        if isinstance(tgt_tokens, list):
            tgt_seq = [item[:, :-1] for item in tgt_tokens]
        else:
            tgt_seq = tgt_tokens[:, :-1]

        seq_probs, pred_embs, *_ = self.decoder(
            tgt_seq=tgt_seq, 
            enc_output=encoder_outputs['enc_output'], 
            category=category,
            tags=encoder_outputs.get(Constants.mapping['attr'][0], None)
            )
        #pred_embs = seq_probs[-1] if isinstance(seq_probs, list) else seq_probs

        if isinstance(seq_probs, list):
            seq_probs = [F.log_softmax(self.tgt_word_prj(item), dim=-1) for item in seq_probs]
        else:
            seq_probs = F.log_softmax(self.tgt_word_prj(seq_probs), dim=-1)


        if bert_embs is not None:
            assert self.bert_embs_to_hidden is not None
            bert_embs = self.bert_embs_to_hidden(bert_embs)

        if self.attribute_predictor is not None:
            pred_attr = self.attribute_predictor(pred_embs)
        else:
            pred_attr = None

        return {Constants.mapping['lang'][0]: seq_probs, Constants.mapping['attr'][0]: encoder_outputs.get(Constants.mapping['attr'][0], None),
                Constants.mapping['dist'][0]: pred_embs, Constants.mapping['dist'][1]: bert_embs,
                Constants.mapping['attr2'][0]: pred_attr
        }


    def forward_ARFormer_rl(self, kwargs):
        feats, category, sample_max, sample_rl = map(
            lambda x: kwargs[x], 
            ["feats", "category", "sample_max", "sample_rl"]
        )
        temperature = kwargs.get('temperature', 1)
        mixer_from = 0
        # either sample_max or sample_rl is true
        assert sample_max ^ sample_rl
        
        # get the encoded video representations
        encoder_outputs = self.encode(feats)
        enc_output = encoder_outputs['enc_output']

        max_len = 16 #self.opt['max_len']
        batch_size, device = enc_output.size(0), enc_output.device

        # use to record the sampled results
        seq = []
        seqLogprobs = []

        # initialize the sequence as the begin-of-sentence token
        tgt_seq = [enc_output.new(batch_size, 1).fill_(Constants.BOS).long()]

        for timestep in range(max_len - 1):
            dec_output, *_ = self.decoder(
                tgt_seq=torch.cat(tgt_seq, dim=1), # [batch_size, timestep+1]
                enc_output=enc_output, 
                category=category,
            )
            # pick the last step
            dec_output = dec_output[-1][:, -1, :] if isinstance(dec_output, list) else dec_output[:, -1, :]
            logprobs = F.log_softmax(self.tgt_word_prj(dec_output), dim=1)

            if sample_max: # Greedy decoding
                sampleLogprobs, it = torch.max(logprobs, 1)
                it = it.view(-1).long()

            if sample_rl:# Sampling from multinomial for self-critical
                if timestep >= mixer_from:
                    prob_prev = torch.exp(torch.div(logprobs, temperature))     # fetch prev distribution (softmax)

                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                    #print(prob_prev.gather(1, it)[:5], prob_prev.max(1)[0][:5])
                    it = it.view(-1).long() # flatten indices for saving in tensor
                else:
                    sampleLogprobs, it = torch.max(logprobs, 1)
                    it = it.view(-1).long()
                
                
            # Replace <end> token (if there is) with 0, i.e., Constants.PAD
            it = it.clone()
            #it[it == Constants.EOS] = 0

            # If all batches predict the <end> token, then stop looping
            unfinished = it.ne(Constants.EOS) if not timestep else (unfinished & it.ne(Constants.EOS)) #it > 0 if not timestep else unfinished * (it > 0)
            #print(unfinished)
            # record
            it = it * unfinished.type_as(it)
            seq.append(it)
            seqLogprobs.append(sampleLogprobs.view(-1))
            
            # update the input of the next step
            tgt_seq.append(it.unsqueeze(1))
            
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return torch.stack(seq, 1), torch.stack(seqLogprobs, 1)


    def forward_LSTM_rl(self, kwargs):
        feats, category, sample_max, sample_rl = map(
            lambda x: kwargs[x], 
            ["feats", "category", "sample_max", "sample_rl"]
        )
        temperature = kwargs.get('temperature', 1)
        mixer_from = 0
        # either sample_max or sample_rl is true
        assert sample_max ^ sample_rl
        
        # get the encoded video representations
        encoder_outputs = self.encode(feats)
        enc_output, enc_hidden = encoder_outputs['enc_output'], encoder_outputs['enc_hidden']

        max_len = 16 #self.opt['max_len']
        batch_size, device = enc_output.size(0), enc_output.device

        # use to record the sampled results
        seq = []
        seqLogprobs = []

        it = enc_output.new(batch_size, 1).fill_(Constants.BOS).long()
        state = self.decoder.init_hidden(enc_hidden)

        for timestep in range(max_len - 1):
            results = self.decoder(
                        it=it, 
                        encoder_outputs=enc_output, 
                        category=category, 
                        decoder_hidden=state, 
                        )
            
            dec_output = results['dec_outputs']
            state = results['dec_hidden']

            logprobs = F.log_softmax(self.tgt_word_prj(dec_output), dim=1)

            if sample_max: # Greedy decoding
                sampleLogprobs, it = torch.max(logprobs, 1)
                it = it.view(-1).long()

            if sample_rl:# Sampling from multinomial for self-critical
                if timestep >= mixer_from:
                    prob_prev = torch.exp(torch.div(logprobs, temperature))     # fetch prev distribution (softmax)

                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                    #print(prob_prev.gather(1, it)[:5], prob_prev.max(1)[0][:5])
                    it = it.view(-1).long() # flatten indices for saving in tensor
                else:
                    sampleLogprobs, it = torch.max(logprobs, 1)
                    it = it.view(-1).long()

            # Replace <end> token (if there is) with 0, i.e., Constants.PAD
            it = it.clone()
            #it[it == Constants.EOS] = 0

            # If all batches predict the <end> token, then stop looping
            unfinished = it.ne(Constants.EOS) if not timestep else (unfinished & it.ne(Constants.EOS)) #it > 0 if not timestep else unfinished * (it > 0)
            #print(unfinished)
            # record
            it = it * unfinished.type_as(it)
            seq.append(it)
            seqLogprobs.append(sampleLogprobs.view(-1))
            
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return torch.stack(seq, 1), torch.stack(seqLogprobs, 1)


    def forward_LSTM(self, kwargs):
        # get the infomation we need from the kwargs
        feats, tgt_tokens, category, opt, taggings, semantics = map(
            lambda x: kwargs[x], 
            ["feats", "tgt_tokens", "category", "opt", "taggings", "semantics"]
        )

        # encode feats
        encoder_outputs = self.encode(feats=feats, semantics=semantics)

        enc_output, enc_hidden = encoder_outputs['enc_output'], encoder_outputs['enc_hidden']

        tmp_tensor = feats if not isinstance(feats, list) else feats[0]

        batch_size = tmp_tensor.size(0)
        state = self.decoder.init_hidden(enc_hidden)
        outputs = []
        attentions = []
        pred_tags = []

        teacher_prob = opt.get('teacher_prob', 1)

        def scheduled(i, sample_mask, item, pre_results):
            if item is None or pre_results is None:
                return None
            if sample_mask.sum() == 0:
                it = item[:, i].clone()
            else:
                sample_ind = sample_mask.nonzero().view(-1)
                it = item[:, i].data.clone()
                prob_prev = torch.exp(pre_results.detach()) # fetch prev distribution: shape Nx(M+1)
                it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))

            return it


        for i in range(tgt_tokens.size(1) - 1):
            
            if i >= 1 and teacher_prob < 1.0: # otherwiste no need to sample
                sample_prob = tmp_tensor.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob > teacher_prob
                it = scheduled(i, sample_mask, tgt_tokens, outputs[-1])
                #tag = scheduled(i, sample_mask, taggings, pred_tags[-1] if len(pred_tags) else None)
                tag = taggings[:, i].clone() if taggings is not None else None
            else:
                it = tgt_tokens[:, i].clone()
                tag = taggings[:, i].clone() if taggings is not None else None
            
            #it = tgt_tokens[:, i].clone()
            results = self.decoder(
                        it=it, 
                        encoder_outputs=enc_output, 
                        category=category, 
                        decoder_hidden=state, 
                        tag=tag
                        )
            
            output = results['dec_outputs']
            state = results['dec_hidden']
            attention = results['weights']
            pred_tag = results.get('pred_tag', None)

            output = F.log_softmax(self.tgt_word_prj(output), dim=1)

            outputs.append(output)
            attentions.append(attention)
            if pred_tag is not None:
                pred_tags.append(pred_tag)

        outputs = torch.stack(outputs, dim=1)
        attentions = torch.stack(attentions, dim=1)
        if len(pred_tags):
            pred_tags = torch.stack(pred_tags, dim=1)
        else:
            pred_tags = None

        final_outputs = encoder_outputs
        final_outputs.update({
                'attentions': attentions,  
                Constants.mapping['lang'][0]: outputs,
                Constants.mapping['tag'][0]: pred_tags
            })
        return final_outputs

    def forward_ENSEMBLE(self, feats, tgt_tokens, category, opt):
        # encode feats
        encoder_outputs = self.encode(feats)
        enc_output, enc_hidden = encoder_outputs['enc_output'], encoder_outputs['enc_hidden']

        tmp_tensor = enc_hidden[0] if not isinstance(enc_hidden[0], tuple) else enc_hidden[0][0]

        batch_size = tmp_tensor.size(0)
        state = self.decoder.init_hidden(enc_hidden)
        outputs = []

        teacher_prob = opt.get('teacher_prob', 1)

        for i in range(tgt_tokens.size(1) - 1):
            
            if i >= 1 and teacher_prob < 1.0: # otherwiste no need to sample
                sample_prob = tmp_tensor.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob > teacher_prob
                if sample_mask.sum() == 0:
                    it = tgt_tokens[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = tgt_tokens[:, i].data.clone()
                    prob_prev = torch.exp(outputs[-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    #it.index_copy_(0, sample_ind, torch.max(outputs[-1], 1)[1].view(-1).index_select(0, sample_ind))
            else:
                it = tgt_tokens[:, i].clone()
            
            it = tgt_tokens[:, i].clone()
            output, state, attention = self.decoder(it, enc_output, category, state)
            
            output = torch.stack(output, dim=1)
            _, length, _ = output.shape
            output = F.log_softmax(self.tgt_word_prj(output.contiguous().view(batch_size * length, -1)), dim=1).view(batch_size, length, -1).mean(1)

            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)

        return {'seq_probs': outputs}

    def forward_beam_decoder(self, kwargs):
        """
            1.  tgt_tokens是描述模型beam search得到的结果 
                [cls] [a] ... [singing] [pad] ... [pad]
            2.  将tgt_tokens输入到beam_decoder中
                得到该句子的置信度logit([cls]标签对应的隐状态映射成1维数据的sigmoid输出)
            3.  ground-truth标签是由单句子的CIDEr METEOR决定
                比如，beam search top-k = 5，令2个指标最高的句子标签为1，其余为0
        """
        tgt_tokens, category, feats = map(
            lambda x: kwargs[x], 
            ["tgt_tokens", "category", "feats"]
        )

        if self.opt.get('bd_load_feats', False):
            encoder_outputs = self.encode(feats)['enc_output']
        else:
            encoder_outputs = None


        logit, *_ = self.beam_decoder(
            tgt_seq=tgt_tokens, 
            category=category,
            enc_output=encoder_outputs
            )

        return {Constants.mapping['beam'][0]: logit}


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Attention(nn.Module):
    def __init__(self, 
        dim_hidden, dim_feats, dim_mid, 
        activation=F.tanh, activation_type='acc', fusion_type='addition'):
        super(Attention, self).__init__()
        assert isinstance(dim_feats, list)
        self.num_feats = len(dim_feats)
        self.dim_hidden = dim_hidden
        self.dim_feats = dim_feats
        self.dim_mid = dim_mid
        self.activation = activation
        self.activation_type = activation_type
        self.fusion_type = fusion_type

        self.linear1_h = nn.Linear(dim_hidden, dim_mid, bias=True)
        for i in range(self.num_feats):
            self.add_module("linear1_f%d"%i, nn.Linear(dim_feats[i], dim_mid, bias=True))
          
        self.linear1_f = []
        for name, module in self.named_children():  
            if 'linear1_f' in name: self.linear1_f.append(module)

        self.linear2_temporal = nn.Linear(dim_mid if fusion_type == 'addition' else dim_mid * 2, 1, bias=False)
        self.mlp = nn.Sequential(
                    nn.Dropout(0.3),
                    #nn.Linear(dim_hidden, dim_hidden * 2),
                    #GELU(),
                    #nn.Dropout(0.3),
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.Tanh(),
                )
        self._init_weights()
        

    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def cal_out(self, linear1_list, linear2, input_list):
        assert isinstance(linear1_list, list) and isinstance(input_list, list)
        assert len(linear1_list) == len(input_list)

        batch_size, seq_len, _ = input_list[-1].size()
        res = []
        for i in range(len(input_list)):
            feat = input_list[i]
            if len(feat.shape) == 2: 
                feat = feat.unsqueeze(1).repeat(1, seq_len, 1)
            linear1_output = linear1_list[i](feat.contiguous().view(batch_size*seq_len, -1))
            res.append(self.activation(linear1_output) if self.activation_type == 'split' else linear1_output)
        
        if self.fusion_type == 'addition':
            output = torch.stack(res).sum(0)
        else:
            output = torch.cat(res, dim=1)
        if self.activation_type != 'split':
            output = self.activation(output)

        output = linear2(output).view(batch_size, seq_len)
        weight = F.softmax(output, dim=1)
        return output, weight

    def forward(self, hidden_state, feats, enhance_feats=None, category=None, t=None):
        """
        feats: [batch_size, seq_len, dim]
        """

        if len(hidden_state.shape) == 1:
            hidden_state = hidden_state.unsqueeze(0)
        if len(hidden_state.shape) == 3 and hidden_state.shape[0] == 1:
            hidden_state = hidden_state.squeeze(0)

        outputs = []
        for i in range(self.num_feats):
            output, _ = self.cal_out(
                [self.linear1_h, self.linear1_f[i]],
                self.linear2_temporal,
                [hidden_state, feats[i]]
                )
            outputs.append(output)
        outputs = torch.stack(outputs, dim=2) # bsz, seq_len, num_feats
        bsz, seq_len, _ = outputs.shape

        weights = F.softmax(outputs, dim=2).view(bsz * seq_len, self.num_feats).unsqueeze(1)        # bsz * seq_len, 1, num_feats
        stacked_feats = torch.stack(feats, dim=2).view(bsz * seq_len, self.num_feats, self.dim_hidden) # bsz * seq_len, num_feats, dim_hidden

        context = torch.bmm(weights, stacked_feats).squeeze(1).view(bsz, seq_len, self.dim_hidden)

        return self.mlp(context), F.softmax(outputs, dim=2)


