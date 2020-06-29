# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """
import json
import logging
import math
import os
import sys

import torch
from torch import nn
import numpy as np
from models.bert_config import BertConfig
import torch.nn.functional as F
import models.Constants as Constants
from torch.nn.parameter import Parameter
def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}


BertLayerNorm = torch.nn.LayerNorm


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq, watch=0):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    if watch != 0 and len_s >= watch:
        assert watch > 0
        tmp = torch.tril(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=-watch)
    else:
        tmp = None

    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    if tmp is not None:
        subsequent_mask += tmp
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]




class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, return_pos=False):
        super(BertEmbeddings, self).__init__()
        '''
        dim_w_hidden = 300
        self.word_embeddings = nn.Embedding(config.vocab_size, dim_w_hidden, padding_idx=Constants.PAD)
        self.prj = nn.Linear(dim_w_hidden, config.dim_hidden, bias=False)
        nn.init.xavier_normal_(self.prj.weight)
        '''

        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim_hidden, padding_idx=Constants.PAD)
        self.position_embeddings = nn.Embedding(config.max_len, config.dim_hidden)
        #self.position_embeddings = PositionalEmbedding(config.max_len, config.dim_hidden)
        self.category_embeddings = nn.Embedding(config.num_category, config.dim_hidden) if config.with_category else None
        self.return_pos = return_pos

        self.LayerNorm = BertLayerNorm(config.dim_hidden, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.return_pos:
            self.pos_LN = BertLayerNorm(config.dim_hidden, eps=config.layer_norm_eps)
            self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.tgt_word_prj = nn.Linear(config.dim_hidden, config.vocab_size, bias=False) if not config.shared_embedding else None
        if self.tgt_word_prj is not None:
            nn.init.xavier_normal_(self.tgt_word_prj.weight)
        else:
            self.tgt_word_prj_bias = Parameter(torch.Tensor(config.vocab_size))
            stdv = 1.0 / math.sqrt(config.vocab_size)
            self.tgt_word_prj_bias.data.uniform_(-stdv, stdv)

        if config.use_tag:
            self.tag2hidden = nn.Linear(config.dim_t, config.dim_hidden)
            nn.init.xavier_normal_(self.tag2hidden.weight)
            #import pickle
            #self.attribute_mapping = torch.LongTensor(pickle.load(open(config.info_corpus, 'rb'))['info']['attribute_mapping'])
        else:
            self.tag2hidden = None
            #self.attribute_mapping = None
        


    def forward(self, input_ids, category=None, position_ids=None, additional_feats=None, tags=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        #words_embeddings = self.prj(self.word_embeddings(input_ids))
        position_embeddings = self.position_embeddings(position_ids)
        if self.category_embeddings is not None:
            assert category is not None
            category_embeddings = self.category_embeddings(category).repeat(1, words_embeddings.size(1), 1)


        if not self.return_pos:
            embeddings = words_embeddings + position_embeddings
            if self.category_embeddings is not None:
                embeddings += category_embeddings
            
            if additional_feats is not None:
                embeddings += additional_feats

            
            #if self.attribute_mapping is not None:
            if self.tag2hidden is not None:
                assert tags is not None
                
                embeddings += self.tag2hidden(tags).unsqueeze(1).repeat(1, embeddings.size(1), 1)

                '''
                self.attribute_mapping = self.attribute_mapping.to(input_ids.device)
                most_possible_attr_ind = tags.topk(15, largest=True, sorted=False)[1]
                most_possible_attr_ind = self.attribute_mapping.unsqueeze(0).repeat(input_ids.size(0), 1).gather(1, most_possible_attr_ind)
                tags_embeddings = self.word_embeddings(most_possible_attr_ind).mean(1)
                embeddings += tags_embeddings.unsqueeze(1).repeat(1, embeddings.size(1), 1)
                '''



            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings
        else:
            embeddings = words_embeddings + position_embeddings
            if self.category_embeddings is not None:
                embeddings += category_embeddings

            if additional_feats is not None:
                embeddings += additional_feats

            embeddings = self.dropout(self.LayerNorm(embeddings))
            position_embeddings = self.pos_dropout(self.pos_LN(position_embeddings))

            return embeddings, position_embeddings
        #return words_embeddings, position_embeddings

    def linear(self, x):
        if self.tgt_word_prj is not None:
            return self.tgt_word_prj(x)
        x = x.matmul(self.word_embeddings.weight.t()) + self.tgt_word_prj_bias# [batch_size, vocab_size]
        return x

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.dim_hidden % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.dim_hidden, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.dim_hidden / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.dim_hidden, self.all_head_size)
        self.key = nn.Linear(config.dim_hidden, self.all_head_size)
        self.value = nn.Linear(config.dim_hidden, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        #self._init_weights()

    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def forward(self, q, k, v, attention_mask, head_mask=None):
        d_k, d_v, n_head = self.attention_head_size, self.attention_head_size, self.num_attention_heads

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.query(q).view(sz_b, len_q, n_head, d_k)
        k = self.key(k).view(sz_b, len_k, n_head, d_k)
        v = self.value(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if attention_mask is not None:
            attention_mask = attention_mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        attention_scores = torch.bmm(q, k.transpose(1, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, -10e6)
        #print("before att:", attention_scores)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        #print("after att:", attention_probs)
        attention_probs = self.dropout(attention_probs)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        outputs = torch.bmm(attention_probs, v)

        outputs = outputs.view(n_head, sz_b, len_q, d_v)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        return (outputs, attention_probs) if self.output_attentions else (outputs,)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.dim_hidden, config.dim_hidden)
        self.LayerNorm = BertLayerNorm(config.dim_hidden, eps=config.layer_norm_eps) if config.with_layernorm else None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        #hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        if self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, pos=False):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pos = pos

    def forward(self, q, k, v, attention_mask, head_mask=None):
        self_outputs = self.self(q, k, v, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], q if not self.pos else v)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.dim_hidden, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        #self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        #hidden_states = self.dropout(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.dim_hidden)
        self.LayerNorm = BertLayerNorm(config.dim_hidden, eps=config.layer_norm_eps) if config.with_layernorm else None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        if self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states)

        return self.dropout(hidden_states)

class BertLayer(nn.Module):
    def __init__(self, config, decoder_layer=False):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.pos_attention = BertAttention(config, pos=True) if (config.pos_attention and decoder_layer) else None
        if decoder_layer:
            self.attend_to_enc_output = BertAttention(config)
        else:
            self.attend_to_enc_output = None

        #self.attend_to_attribute = BertAttention(config) if config.use_tag else None

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        

    def forward(self, hidden_states, enc_output=None, non_pad_mask=None, attention_mask=None, head_mask=None, attend_to_enc_output_mask=None, position_embeddings=None, 
        attributes=None, attend_to_attribute_mask=None):
        attention_outputs = self.attention(hidden_states, hidden_states, hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        if non_pad_mask is not None:
            attention_output *= non_pad_mask

        embs = attention_output.clone()

        if self.pos_attention is not None:
            assert position_embeddings is not None
            attention_outputs = self.pos_attention(position_embeddings, position_embeddings, attention_output, attention_mask, head_mask)
            attention_output = attention_outputs[0]
            if non_pad_mask is not None:
                attention_output *= non_pad_mask

        if self.attend_to_enc_output is not None:
            assert attend_to_enc_output_mask is not None
            assert enc_output is not None

            attention_outputs = self.attend_to_enc_output(attention_output, enc_output, enc_output, attend_to_enc_output_mask, head_mask)#enc_output[:, -1, :])

            attention_output = attention_outputs[0]
            if non_pad_mask is not None:
                attention_output *= non_pad_mask

        '''
        if self.attend_to_attribute is not None:
            assert attend_to_attribute_mask is not None
            assert attributes is not None
            attention_outputs = self.attend_to_enc_output(attention_output, attributes, attributes, attend_to_attribute_mask, head_mask)#enc_output[:, -1, :])
            attention_output = attention_outputs[0]
            if non_pad_mask is not None:
                attention_output *= non_pad_mask
        '''

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        #layer_output = attention_output
        if non_pad_mask is not None:
            layer_output *= non_pad_mask

        # multitask_attribute
        #print(layer_output.shape, non_pad_mask.shape)
        embs = layer_output.sum(1) / non_pad_mask.sum(1)

        outputs = (layer_output,embs,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

'''

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, return_pos=False):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim_hidden, padding_idx=Constants.PAD)
        self.position_embeddings = nn.Embedding(config.max_len, config.dim_hidden)
        #self.position_embeddings = PositionalEmbedding(config.max_len, config.dim_hidden)
        self.category_embeddings = nn.Embedding(config.num_category, config.dim_hidden) if config.with_category else None
        self.return_pos = return_pos
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids, category=None, position_ids=None, additional_feats=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        if self.category_embeddings is not None:
            assert category is not None
            category_embeddings = self.category_embeddings(category).repeat(1, words_embeddings.size(1), 1)


        if not self.return_pos:
            embeddings = words_embeddings + position_embeddings
            if self.category_embeddings is not None:
                embeddings += category_embeddings
            
            if additional_feats is not None:
                embeddings += additional_feats
            embeddings = self.dropout(embeddings)
            return embeddings
        else:
            embeddings = words_embeddings
            if self.category_embeddings is not None:
                embeddings += category_embeddings
            embeddings = self.dropout(embeddings)
            position_embeddings = self.dropout(position_embeddings)

            return embeddings, position_embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.dim_hidden % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.dim_hidden, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.dim_hidden / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.dim_hidden, self.all_head_size)
        self.key = nn.Linear(config.dim_hidden, self.all_head_size)
        self.value = nn.Linear(config.dim_hidden, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        #self._init_weights()

    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def forward(self, q, k, v, attention_mask, head_mask=None):
        d_k, d_v, n_head = self.attention_head_size, self.attention_head_size, self.num_attention_heads

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.query(q).view(sz_b, len_q, n_head, d_k)
        k = self.key(k).view(sz_b, len_k, n_head, d_k)
        v = self.value(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if attention_mask is not None:
            attention_mask = attention_mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        attention_scores = torch.bmm(q, k.transpose(1, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, -10e6)
        #print("before att:", attention_scores)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        #print("after att:", attention_probs)
        attention_probs = self.dropout(attention_probs)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        outputs = torch.bmm(attention_probs, v)

        outputs = outputs.view(n_head, sz_b, len_q, d_v)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        return (outputs, attention_probs) if self.output_attentions else (outputs,)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.dim_hidden, config.dim_hidden)     
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        return self.dropout(self.dense(hidden_states))


class BertAttention(nn.Module):
    def __init__(self, config, att_type):
        super(BertAttention, self).__init__()
        self.LayerNorm = BertLayerNorm(config.dim_hidden, eps=config.layer_norm_eps) if config.with_layernorm else None
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

        assert att_type in ['self', 'eo', 'pos']
        self.att_type = att_type

    def forward(self, hidden_states, another_input, attention_mask, head_mask=None):
        residual = hidden_states.clone()
        
        hidden_states = self.LayerNorm(hidden_states) if self.LayerNorm is not None else hidden_states

        if self.att_type == 'self':    
            self_outputs = self.self(hidden_states, hidden_states, hidden_states, attention_mask, head_mask)
        elif self.att_type == 'eo':
            self_outputs = self.self(hidden_states, another_input, another_input, attention_mask, head_mask)
        else:
            another_input = self.LayerNorm(another_input) if self.LayerNorm is not None else another_input
            self_outputs = self.self(another_input, another_input, hidden_states, attention_mask, head_mask)

        
        attention_output = self.output(self_outputs[0]) + residual
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class FeedFoward(nn.Module):
    def __init__(self, config):
        super(FeedFoward, self).__init__()
        self.LayerNorm = BertLayerNorm(config.dim_hidden, eps=config.layer_norm_eps) if config.with_layernorm else None
        self.dense = nn.Linear(config.dim_hidden, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.dense2 = nn.Linear(config.intermediate_size, config.dim_hidden)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        #self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        residual = hidden_states.clone()
        hidden_states = self.dense(hidden_states if self.LayerNorm is None else self.LayerNorm(hidden_states))
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + residual


class BertLayer(nn.Module):
    def __init__(self, config, decoder_layer=False):
        super(BertLayer, self).__init__()
        residual = config.residual
        self.attention = BertAttention(config, att_type='self')
        self.pos_attention = BertAttention(config, att_type='pos') if (config.pos_attention and decoder_layer) else None
        if decoder_layer:
            self.attend_to_enc_output = BertAttention(config, att_type='eo')
        else:
            self.attend_to_enc_output = None
        self.feedfoward = FeedFoward(config)

    def forward(self, hidden_states, enc_output=None, non_pad_mask=None, attention_mask=None, head_mask=None, attend_to_enc_output_mask=None, position_embeddings=None):
        attention_outputs = self.attention(hidden_states, None, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        if non_pad_mask is not None:
            attention_output *= non_pad_mask

        if self.pos_attention is not None:
            assert position_embeddings is not None
            attention_outputs = self.pos_attention(attention_output, position_embeddings, attention_mask, head_mask)
            attention_output = attention_outputs[0]
            if non_pad_mask is not None:
                attention_output *= non_pad_mask

        if self.attend_to_enc_output is not None:
            assert attend_to_enc_output_mask is not None
            assert enc_output is not None

            attention_outputs = self.attend_to_enc_output(attention_output, enc_output, attend_to_enc_output_mask, head_mask)#enc_output[:, -1, :])

            attention_output = attention_outputs[0]
            if non_pad_mask is not None:
                attention_output *= non_pad_mask

        layer_output = self.feedfoward(attention_output)
        #layer_output = attention_output
        if non_pad_mask is not None:
            layer_output *= non_pad_mask
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs
'''
def resampling(source, tgt_tokens):
    pad_mask = tgt_tokens.eq(Constants.PAD)
    length = (1 - pad_mask).sum(-1)
    bsz, seq_len = tgt_tokens.shape

    all_idx = []
    scale = source.size(1) / length.float()
    for i in range(bsz):
        idx = (torch.arange(0, seq_len, device=tgt_tokens.device).float() * scale[i].repeat(seq_len)).long()
        max_idx = tgt_tokens.new(seq_len).fill_(source.size(1) - 1)
        idx = torch.where(idx < source.size(1), idx, max_idx)
        all_idx.append(idx)
    all_idx = torch.stack(all_idx, dim=0).unsqueeze(2).repeat(1, 1, source.size(2))
    return source.gather(1, all_idx)
    '''

        for j in range(seq_len):
            idx = min(source.size(1) - 1, int(j * scale[i]))
            tmp.append(source[i, idx, :])
        res.append(torch.stack(tmp, dim=0))

    torch.arange(7, 12, device=beam_starts.device)
    return torch.stack(res, dim=0)
    
       res = []
        scale = source.size(1) / tgt_len
        []

    res = []
    scale = source.size(1) / tgt_len
    for i in range(tgt_len):
        idx = int(i * scale)
        res.append(source[:, idx, :])
    return torch.stack(res, dim=1)
    '''


class EmptyObject(object):
    def __init__(self):
        pass

def dict2obj(dict):
    obj = EmptyObject()
    obj.__dict__.update(dict)
    return obj

class BertDecoder(nn.Module):
    def __init__(self, config=BertConfig(), embedding=None):
        super(BertDecoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding = BertEmbeddings(config, return_pos=True if config.pos_attention else False) if embedding is None else embedding
        self.layer = nn.ModuleList([BertLayer(config, decoder_layer=True) for _ in range(config.num_hidden_layers_decoder)])
        self.pos_attention = config.pos_attention
        self.enhance_input = config.enhance_input
        self.watch = config.watch
        self.decoder_type = config.decoder_type

        '''
        if config.use_tag:
            import pickle
            #self.attribute_mapping = torch.LongTensor(pickle.load(open(config.info_corpus, 'rb'))['info']['attribute_mapping'])
            self.attribute_topk = 15
            self.attribute_embs = nn.Embedding(config.dim_t, config.dim_hidden)
        else:
            self.attribute_mapping = None
            self.attribute_embs = None
        '''


    def _init_embedding(self, weight, option={}):
        self.embedding.word_embeddings.weight.data.copy_(weight.data)
        if not option.get('train_emb', True):
            for p in self.embedding.word_embeddings.parameters():
                p.requires_grad = False

    def forward(self, tgt_seq, enc_output, category, signals=None, tags=None):
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]
        all_attentions = ()

        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        if self.decoder_type == 'NARFormer':
            slf_attn_mask = slf_attn_mask_keypad
            #tmp = torch.tril(torch.ones((tgt_seq.size(1), tgt_seq.size(1)), dtype=torch.uint8), diagonal=0)
            #tmp2 = torch.triu(torch.ones((tgt_seq.size(1), tgt_seq.size(1)), dtype=torch.uint8), diagonal=0)
            #slf_attn_mask = (slf_attn_mask + (tmp & tmp2).to(tgt_seq.device)).gt(0)
        else:
            slf_attn_mask_subseq = get_subsequent_mask(tgt_seq, watch=self.watch)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        non_pad_mask = get_non_pad_mask(tgt_seq)

        #print(slf_attn_mask)
        #print(non_pad_mask)

        src_seq = torch.ones(enc_output.size(0), enc_output.size(1)).to(enc_output.device)
        attend_to_enc_output_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        #print(slf_attn_mask[0])
        #print(attend_to_enc_output_mask[0])
        #print(slf_attn_mask)

        if self.enhance_input == 1:
            additional_feats = resampling(enc_output, tgt_seq)
        elif self.enhance_input == 2:
            additional_feats = enc_output.mean(1).unsqueeze(1).repeat(1, tgt_seq.size(1), 1)
        else:
            additional_feats = None

        if signals is not None:
            additional_feats = signals if additional_feats is None else (additional_feats + signals)

        if self.pos_attention:
            hidden_states, position_embeddings = self.embedding(tgt_seq, category=category)
        else:
            hidden_states = self.embedding(tgt_seq, additional_feats=additional_feats, category=category, tags=tags)
            position_embeddings = None

        '''
        if self.attribute_embs is not None:
            attend_to_attribute_mask = get_attn_key_pad_mask(seq_k=torch.ones(enc_output.size(0), self.attribute_topk).to(enc_output.device), seq_q=tgt_seq)
            assert tags is not None            
            #self.attribute_mapping = self.attribute_mapping.to(tgt_seq.device)
            most_possible_attr_ind = tags.topk(self.attribute_topk, largest=True, sorted=False)[1]
            #most_possible_attr_ind = self.attribute_mapping.unsqueeze(0).repeat(tgt_seq.size(0), 1).gather(1, most_possible_attr_ind)
            #attributes = self.embedding.word_embeddings(most_possible_attr_ind)
            attributes = self.attribute_embs(most_possible_attr_ind)
        else:
        '''
        attributes = None
        attend_to_attribute_mask = None


        res = []
        for i, layer_module in enumerate(self.layer):
            if not i:
                input_ = hidden_states
            else:
                input_ = layer_outputs[0]# + hidden_states
            layer_outputs = layer_module(input_, enc_output=enc_output, 
                attention_mask=slf_attn_mask, attend_to_enc_output_mask=attend_to_enc_output_mask, position_embeddings=position_embeddings, non_pad_mask=non_pad_mask,
                attributes=attributes, attend_to_attribute_mask=attend_to_attribute_mask
                )

            res.append(layer_outputs[0])
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            embs = layer_outputs[1]

        res = [res[-1]]
        outputs = (res,embs,)

        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BasicAttention(nn.Module):
    def __init__(self, 
        dim_hidden, dim_feats, dim_mid, 
        activation=torch.tanh, activation_type='acc', fusion_type='addition'):
        super(BasicAttention, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_feats = dim_feats
        self.dim_mid = dim_mid
        self.activation = activation
        self.activation_type = activation_type
        self.fusion_type = fusion_type

        self.linear1_h = nn.Linear(dim_hidden, dim_mid, bias=True)
        self.linear1_f = nn.Linear(dim_feats, dim_mid, bias=True)
        self.linear2_temporal = nn.Linear(dim_mid if fusion_type == 'addition' else dim_mid * 2, 1, bias=False)
        self._init_weights()
        

    def _init_weights(self):
        for module in self.children():
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

    def forward(self, hidden_state, feats):
        """
        hidden_state: [batch_size, seq_len1, dim]
        feats: [batch_size, seq_len2, dim]
        """

        b, seq_len, _ = hidden_state.shape
        seq_len2 = feats.size(1)
        hidden_state = hidden_state.contiguous().view(b * seq_len, -1)
        
        _, weight = self.cal_out(
            [self.linear1_h, self.linear1_f],
            self.linear2_temporal,
            [hidden_state, feats.unsqueeze(1).repeat(1, seq_len, 1, 1).contiguous().view(b * seq_len, seq_len2, -1)]
            )

        # weight: [batch_size * seq_len, seq_len2]
        weight = weight.view(b, seq_len, seq_len2)
        context = torch.bmm(weight, feats)     # [batch_size, seq_len, dim]

        return context, weight


class EncoderEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, feats_size):
        super(EncoderEmbeddings, self).__init__()
        self.feats_embeddings = nn.Linear(feats_size, config.dim_hidden)
        self.position_embeddings = nn.Embedding(config.n_frames, config.dim_hidden)
        self.LayerNorm = BertLayerNorm(config.dim_hidden, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.feats_activation = ACT2FN[config.feat_act]


    def forward(self, feats, position_ids=None):
        seq_length = feats.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=feats.device)
            position_ids = position_ids.unsqueeze(0).repeat(feats.size(0), 1)

        feats_embeddings = self.feats_activation(self.feats_embeddings(feats))
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = feats_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEncoder(nn.Module):
    def __init__(self, feats_size, config=BertConfig()):
        super(BertEncoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.emb = EncoderEmbeddings(config, feats_size)
        self.layer = nn.ModuleList([BertLayer(config, decoder_layer=False) for _ in range(config.num_hidden_layers_encoder)])

    def forward(self, feats):
        all_attentions = ()
        if isinstance(feats, list):
            assert len(feats) == 1
            f = feats[0]
        else:
            f = feats
        hidden_states = self.emb(f)

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)


        outputs = (hidden_states, hidden_states.mean(1))

        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class ARDecoder_with_attribute_generation(nn.Module):
    def __init__(self, config):
        super(ARDecoder_with_attribute_generation, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.bert = BertDecoder(config)
        self.embedding = self.bert.embedding

    def forward_(self, tgt_seq, enc_output, category, tags=None):
        bsz, seq_len = tgt_seq.shape
        seq_probs, embs, *_ = self.bert(tgt_seq, enc_output, category, tags=tags)
        seq_probs = seq_probs[0]
        return seq_probs, embs

    def forward(self, tgt_seq, enc_output, category, tags=None):
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]

        if isinstance(tgt_seq, list):
            assert len(tgt_seq) == 2
            seq_probs1, _ = self.forward_(tgt_seq[0], enc_output, category, tags=tags)
            seq_probs2, embs = self.forward_(tgt_seq[1], enc_output, category, tags=tags)        
            outputs = ([seq_probs1, seq_probs2],embs,)
        else:
            seq_probs, embs = self.forward_(tgt_seq, enc_output, category, tags=tags)
            outputs = ([seq_probs],embs,)
        return outputs


class ARDecoder(nn.Module):
    def __init__(self, config):
        super(ARDecoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.bert = BertDecoder(config)
        self.embedding = self.bert.embedding

    def forward(self, tgt_seq, enc_output, category, tags=None):
        if self.training:
            assert isinstance(tgt_seq, list)
            assert len(tgt_seq) == 2
            seq_probs1, *_ = self.bert(tgt_seq[0], enc_output, category, tags=tags, causal=False)
            seq_probs1 = seq_probs1[0]
            seq_probs2, *_ = self.bert(tgt_seq[1], enc_output, category, tags=tags)
            seq_probs2 = seq_probs2[0]
            return ([seq_probs1, seq_probs2],)
        else:
            return self.bert(tgt_seq[1], enc_output, category, tags=tags)
        


def enlarge(info, beam_size):
    bsz, *rest_shape = info.shape
    if len(rest_shape) == 2:
        return info.unsqueeze(1).repeat(1, beam_size, 1, 1).view(bsz * beam_size, *rest_shape)
    return info.unsqueeze(1).repeat(1, beam_size, 1).view(bsz * beam_size, *rest_shape)

class NVADecoder(nn.Module):
    def __init__(self, config):
        super(NVADecoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.embedding = BertEmbeddings(config, return_pos=True if config.pos_attention else False)
        self.decoder1 = BertDecoder(config=config, embedding=self.embedding)
        self.decoder2 = BertDecoder(config=config, embedding=self.embedding)

        self.num_signal = len(config.demand)
        self.signal_embedding = nn.Embedding(self.num_signal, config.dim_hidden)

    def forward_decoder1(self, tgt_seq, enc_output, category):
        # tgt_seq: [batch_size * num_signal, seq_len]
        # enc_output: [batch_size, n_frames, dim_hidden]
        bsz, seq_len = tgt_seq.shape
        bsz = bsz // self.num_signal
        signals = torch.cat([tgt_seq.new(bsz, seq_len).fill_(i).long() for i in range(self.num_signal)], dim=0) # [batch_size * num_signal, seq_len]
        signals = self.signal_embedding(signals)

        seq_probs, *_ = self.decoder1(
            tgt_seq, 
            enlarge(enc_output, self.num_signal), 
            enlarge(category, self.num_signal), 
            signals=signals
        )
        seq_probs = seq_probs[0]
        return seq_probs

    def forward_decoder2(self, tgt_seq, enc_output, category):
        seq_probs, *_ = self.decoder2(tgt_seq, enc_output, category)
        seq_probs = seq_probs[0]
        return seq_probs

    def forward(self, tgt_seq, enc_output, category):
        assert isinstance(tgt_seq, list)
        assert len(tgt_seq) == 2
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]
        seq_probs1 = self.forward_decoder1(tgt_seq[0], enc_output, category)
        seq_probs2 = self.forward_decoder2(tgt_seq[1], enc_output, category)
        outputs = ([seq_probs1, seq_probs2],)
        return outputs


def to_sentence(hyp, vocab, break_words=[Constants.PAD], skip_words=[]):
    sent = []
    for word_id in hyp:
        if word_id in skip_words:
            continue
        if word_id in break_words:
            break
        sent.append(vocab[word_id])
    return ' '.join(sent)

from tqdm import tqdm
class DirectDecoder(nn.Module):
    def __init__(self, config):
        super(DirectDecoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.embedding = BertEmbeddings(config, return_pos=True if config.pos_attention else False)
        self.decoder1 = BertDecoder(config=config, embedding=self.embedding)
        self.decoder2 = BertDecoder(config=config, embedding=self.embedding)
        self.masking_ratio = config.masking_ratio
        self.minimun_masking_ratio = 0.2
        self.select_out_ratio = config.select_out_ratio

    def forward_decoder1(self, tgt_seq, enc_output, category):
        seq_probs, *_ = self.decoder1(tgt_seq, enc_output, category)
        seq_probs = seq_probs[0]
        return seq_probs

    def forward_decoder2(self, tgt_seq, enc_output, category):
        seq_probs, *_ = self.decoder2(tgt_seq, enc_output, category)
        seq_probs = seq_probs[0]
        return seq_probs

    def step(self, seq_probs, pad_mask, tgt_word_prj):
        out = tgt_word_prj(seq_probs)
        probs = F.softmax(out, dim=-1) #[batch_size, max_len, vocab_size]
        token_probs, token = probs.max(dim=-1)
        token[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0
        return token, token_probs


    def forward(self, tgt_seq, enc_output, category, tgt_word_prj, vocab):
        assert isinstance(tgt_seq, list)
        assert len(tgt_seq) == 2
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]

        pad_mask = tgt_seq[0].eq(Constants.PAD)
        seq_lens = tgt_seq[0].size(1) - pad_mask.sum(dim=1)

        possibility = torch.rand(tgt_seq[0].size(0)).to(tgt_seq[0].device)
        ind = (possibility > 2*(self.masking_ratio - self.minimun_masking_ratio))
        possibility += 2*self.minimun_masking_ratio
        possibility[ind] = 2*self.masking_ratio
        ratio = possibility / 2

        num_mask = (seq_lens.float() * ratio).long()


        seq_probs1 = self.forward_decoder1(tgt_seq[0], enc_output, category)

        #tqdm.write(" Output 1: " + to_sentence(self.step(seq_probs1, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        #tqdm.write(" Input  2: " + to_sentence(tgt_seq[1][0].tolist(), vocab))

        if self.select_out_ratio:
            possibility = torch.rand(*tgt_seq[1].shape)
            possibility[pad_mask] = 1.0
            select_out_ind = (possibility < self.select_out_ratio)
            input_ = tgt_seq[1].clone()
            input_[select_out_ind] = self.step(seq_probs1, pad_mask, tgt_word_prj)[0][select_out_ind]
            #tqdm.write(" Input 22: " + to_sentence(input_[0].tolist(), vocab))
        else:
            input_ = tgt_seq[1]



        seq_probs2 = self.forward_decoder2(input_, enc_output, category)
        
        
        token, token_probs = self.step(seq_probs2, pad_mask, tgt_word_prj)
        #tqdm.write(" Output 2: " + to_sentence(token[0].tolist(), vocab))
        mask_ind = self.select_worst(token_probs, num_mask)
        token[mask_ind] = Constants.MASK
        #tqdm.write(" Input  3: " + to_sentence(token[0].tolist(), vocab))

        seq_probs3 = self.forward_decoder2(token, enc_output, category)
        #tqdm.write(" Output 3: " + to_sentence(self.step(seq_probs3, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        #tqdm.write("--------------------------------------")
        outputs = ([seq_probs1, seq_probs2, seq_probs3],)
        return outputs

    def select_worst(self, token_probs, num_mask):
        masks = torch.zeros(*token_probs.shape, device=token_probs.device)
        for i in range(masks.size(0)):
            ind = token_probs[i, :].topk(max(1, num_mask[i]), largest=False, sorted=False)[1]
            masks[i, ind] = 1
        return masks.byte()


class APDecoder(nn.Module):
    def __init__(self, config):
        super(APDecoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.embedding = BertEmbeddings(config, return_pos=True if config.pos_attention else False)
        self.decoder1 = BertDecoder(config=config, embedding=self.embedding)
        self.decoder2 = BertDecoder(config=config, embedding=self.embedding)

    def forward_decoder1(self, tgt_seq, enc_output, category):
        seq_probs, *_ = self.decoder1(tgt_seq, enc_output, category)
        seq_probs = seq_probs[0]
        return seq_probs

    def forward_decoder2(self, tgt_seq, enc_output, category):
        seq_probs, *_ = self.decoder2(tgt_seq, enc_output, category)
        seq_probs = seq_probs[0]
        return seq_probs

    def forward(self, tgt_seq, enc_output, category, tgt_word_prj):
        assert isinstance(tgt_seq, list)
        assert len(tgt_seq) == 2
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]
        seq_probs1 = self.forward_decoder1(tgt_seq[0], enc_output, category)

        out = tgt_word_prj(seq_probs1)
        probs = F.softmax(out, dim=-1)
        probs[:, :, Constants.MASK] = 0

        _, decoder2_input = probs.max(dim=-1)

        seq_probs2 = self.forward_decoder2(decoder2_input, enc_output, category)
        outputs = ([seq_probs1, seq_probs2],)
        return outputs



class SignalDecoder(nn.Module):
    def __init__(self, config):
        super(SignalDecoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.bert = BertDecoder(config)
        self.embedding = self.bert.embedding

        self.signal_embedding = nn.Embedding(2, config.dim_hidden) #if not config.no_signal else None

        self.masking_ratio = config.masking_ratio
        self.minimun_masking_ratio = 0.2
        self.select_out_ratio = config.select_out_ratio
        self.random = np.random.RandomState(0)

    def forward_(self, tgt_seq, enc_output, category, signal):
        bsz, seq_len = tgt_seq.shape
        assert signal in [0, 1]

        if self.signal_embedding is None:
            signals = None
        else:
            signals = tgt_seq.new(bsz, seq_len).fill_(signal).long() # [batch_size, seq_len]
            signals = self.signal_embedding(signals)

        seq_probs, *_ = self.bert(tgt_seq, enc_output, category, signals=signals)
        seq_probs = seq_probs[0]
        return seq_probs

    def step(self, seq_probs, pad_mask, tgt_word_prj):
        out = tgt_word_prj(seq_probs)
        probs = F.softmax(out, dim=-1) #[batch_size, max_len, vocab_size]
        token_probs, token = probs.max(dim=-1)
        token[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0
        return token, token_probs


    def forward(self, tgt_seq, enc_output, category, tgt_word_prj, vocab):
        assert isinstance(tgt_seq, list)
        assert len(tgt_seq) == 2
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]

        pad_mask = tgt_seq[0].eq(Constants.PAD)
        bsz, seq_len = tgt_seq[0].size()
        device = tgt_seq[0].device
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        possibility = torch.rand(bsz).to(device)
        ind = (possibility > 2*(self.masking_ratio - self.minimun_masking_ratio))
        possibility += 2*self.minimun_masking_ratio
        possibility[ind] = 2*self.masking_ratio
        ratio = possibility / 2
        
        #ratio = torch.rand(tgt_seq[0].size(0)).to(tgt_seq[0].device)
        
        num_mask = (seq_lens.float() * ratio).long()


        #tqdm.write(" Input  1: " + to_sentence(tgt_seq[0][0].tolist(), vocab))
        seq_probs1 = self.forward_(tgt_seq[0], enc_output, category, signal=0)
        #tqdm.write(" Output 1: " + to_sentence(self.step(seq_probs1, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        #tqdm.write(" Input  2: " + to_sentence(tgt_seq[1][0].tolist(), vocab))

        if self.select_out_ratio:
            possibility = torch.rand(*tgt_seq[1].shape)
            possibility[pad_mask] = 1.0
            select_out_ind = (possibility < self.select_out_ratio)
            input_ = tgt_seq[1].clone()
            input_[select_out_ind] = self.step(seq_probs1, pad_mask, tgt_word_prj)[0][select_out_ind]
            #tqdm.write(" Input 22: " + to_sentence(input_[0].tolist(), vocab))
        else:
            input_ = tgt_seq[1]

        seq_probs2 = self.forward_(input_, enc_output, category, signal=1)
        
        token, token_probs = self.step(seq_probs2, pad_mask, tgt_word_prj)
        #tqdm.write(" Output 2: " + to_sentence(token[0].tolist(), vocab))
        
        mask_ind = self.select_worst(token_probs, num_mask)
        #mask_ind = self.select_random(token_probs, pad_mask)

        token[mask_ind] = Constants.MASK
        #tqdm.write(" Input  3: " + to_sentence(token[0].tolist(), vocab))

        seq_probs3 = self.forward_(token, enc_output, category, signal=1)
        #tqdm.write(" Output 3: " + to_sentence(self.step(seq_probs3, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        #tqdm.write("--------------------------------------")
        outputs = ([seq_probs1, seq_probs2, seq_probs3],)
        return outputs

    def select_worst(self, token_probs, num_mask):
        

        masks = torch.zeros(*token_probs.shape, device=token_probs.device)
        for i in range(masks.size(0)):
            ind = token_probs[i, :].topk(max(1, num_mask[i]), largest=False, sorted=False)[1]
            masks[i, ind] = 1
        return masks.byte()

    def select_random(self, token_probs, pad_mask):
        bsz, seq_len = token_probs.size()
        possibility = torch.rand(bsz, seq_len).to(token_probs.device)
        possibility[pad_mask] = 1.0

        '''
        tmp = torch.rand(bsz).to(token_probs.device)
        ind = (tmp > 2*(self.masking_ratio - self.minimun_masking_ratio))
        tmp += 2*self.minimun_masking_ratio
        tmp[ind] = 2*self.masking_ratio
        ratio = tmp / 2
        ratio = ratio.unsqueeze(1).repeat(1, seq_len)
        '''
        ratio = torch.rand(bsz).to(token_probs.device).unsqueeze(1).repeat(1, seq_len)
        return (possibility < ratio).byte()


class Signal2Decoder(nn.Module):
    def __init__(self, config):
        super(Signal2Decoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.bert = BertDecoder(config)
        self.embedding = self.bert.embedding
        self.signal_embedding = nn.Embedding(2, config.dim_hidden)

        self.masking_ratio = config.masking_ratio
        self.minimun_masking_ratio = 0.2
        self.select_out_ratio = config.select_out_ratio

    def forward_(self, tgt_seq, enc_output, category, signal):
        bsz, seq_len = tgt_seq.shape
        assert signal in [0, 1]

        signals = tgt_seq.new(bsz, seq_len).fill_(signal).long() # [batch_size, seq_len]
        signals = self.signal_embedding(signals)

        seq_probs, *_ = self.bert(tgt_seq, enc_output, category, signals=signals)
        seq_probs = seq_probs[0]
        return seq_probs

    def step(self, seq_probs, pad_mask, tgt_word_prj):
        out = tgt_word_prj(seq_probs)
        probs = F.softmax(out, dim=-1) #[batch_size, max_len, vocab_size]
        token_probs, token = probs.max(dim=-1)
        token[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0
        return token, token_probs


    def forward(self, tgt_seq, enc_output, category, tgt_word_prj, vocab):
        assert isinstance(tgt_seq, list)
        assert len(tgt_seq) == 2
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]

        pad_mask = tgt_seq[0].eq(Constants.PAD)
        seq_lens = tgt_seq[0].size(1) - pad_mask.sum(dim=1)

        possibility = torch.rand(tgt_seq[0].size(0)).to(tgt_seq[0].device)
        ind = (possibility > 2*(self.masking_ratio - self.minimun_masking_ratio))
        possibility += 2*self.minimun_masking_ratio
        possibility[ind] = 2*self.masking_ratio
        ratio = possibility / 2

        num_mask = (seq_lens.float() * ratio).long()

        seq_probs1 = self.forward_(tgt_seq[0], enc_output, category, signal=0)
        #tqdm.write(" Output 1: " + to_sentence(self.step(seq_probs1, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        #tqdm.write(" Input  2: " + to_sentence(tgt_seq[1][0].tolist(), vocab))

        if self.select_out_ratio:
            possibility = torch.rand(*tgt_seq[1].shape)
            possibility[pad_mask] = 1.0
            select_out_ind = (possibility < self.select_out_ratio)
            input_ = tgt_seq[1].clone()
            input_[select_out_ind] = self.step(seq_probs1, pad_mask, tgt_word_prj)[0][select_out_ind]
            #tqdm.write(" Input 22: " + to_sentence(input_[0].tolist(), vocab))
        else:
            input_ = tgt_seq[1]

        seq_probs2 = self.forward_(input_, enc_output, category, signal=1)
        
        outputs = ([seq_probs1, seq_probs2],)
        return outputs

    def select_worst(self, token_probs, num_mask):
        masks = torch.zeros(*token_probs.shape, device=token_probs.device)
        for i in range(masks.size(0)):
            ind = token_probs[i, :].topk(max(1, num_mask[i]), largest=False, sorted=False)[1]
            masks[i, ind] = 1
        return masks.byte()


class Signal3Decoder_ori(nn.Module):
    def __init__(self, config):
        super(Signal3Decoder_ori, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.bert = BertDecoder(config)
        self.embedding = self.bert.embedding
        self.signal_embedding = nn.Embedding(3, config.dim_hidden) if not config.no_signal else None

        self.masking_ratio = config.masking_ratio
        self.minimun_masking_ratio = 0.2
        self.select_out_ratio = config.select_out_ratio

        self.visual_tag = config.visual_tag
        self.nonvisual_tag = config.nonvisual_tag
        self.revision_tag = config.revision_tag

    def forward_(self, tgt_seq, enc_output, category, signal):
        bsz, seq_len = tgt_seq.shape
        assert signal in [0, 1, 2]

        if self.signal_embedding is None:
            signals = None
        else:
            signals = tgt_seq.new(bsz, seq_len).fill_(signal).long() # [batch_size, seq_len]
            signals = self.signal_embedding(signals)

        seq_probs, *_ = self.bert(tgt_seq, enc_output, category, signals=signals)
        seq_probs = seq_probs[0]
        return seq_probs

    def step(self, seq_probs, pad_mask, tgt_word_prj, tag_replace=None):
        out = tgt_word_prj(seq_probs)
        probs = F.softmax(out, dim=-1) #[batch_size, max_len, vocab_size]
        token_probs, token = probs.max(dim=-1)
        token[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0

        if tag_replace is not None:
            source, target = tag_replace
            ind = token.eq(source)
            token[ind] = target
            copy_ = token_probs.clone()
            token_probs[ind] = 0.0
            return token, token_probs, copy_
        return token, token_probs

    def forward(self, tgt_seq, enc_output, category, tgt_word_prj, vocab):
        assert isinstance(tgt_seq, list)
        assert len(tgt_seq) == 3
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]

        pad_mask = tgt_seq[0].eq(Constants.PAD)

        seq_probs1 = self.forward_(tgt_seq[0], enc_output, category, signal=0)
        #tqdm.write(" Input  1: " + to_sentence(tgt_seq[0][0].tolist(), vocab))
        #tqdm.write(" Output 1: " + to_sentence(self.step(seq_probs1, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        seq_probs2 = self.forward_(tgt_seq[1], enc_output, category, signal=1)
        #tqdm.write(" Input  2: " + to_sentence(tgt_seq[1][0].tolist(), vocab))
        #tqdm.write(" Output 2: " + to_sentence(self.step(seq_probs2, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        
        token1, token_probs1, copy_1 = self.step(seq_probs1, pad_mask, tgt_word_prj, tag_replace=[self.visual_tag, self.revision_tag])
        token2, token_probs2, copy_2 = self.step(seq_probs2, pad_mask, tgt_word_prj, tag_replace=[self.nonvisual_tag, self.revision_tag])

        ind_blank = token1.eq(self.visual_tag) & token2.eq(self.nonvisual_tag)
        ind = token_probs2 > token_probs1
        token1[ind] = token2[ind]
        token_probs1[ind] = token_probs2[ind]
        token_probs1[ind_blank] = torch.max(copy_1[ind_blank], copy_2[ind_blank])


        #tqdm.write(" Fusion  : " + to_sentence(token1[0].tolist(), vocab))
        #fused_input = self.fuse_pred_gt(token1, tgt_seq[2], pad_mask)
        #mask_ind = self.select_random(token_probs1, pad_mask)
        #fused_input[mask_ind] = self.revision_tag

        
        mask_ind = self.select_worst(token_probs1, pad_mask)
        token1[mask_ind] = self.revision_tag

        #tqdm.write(" Input  3: " + to_sentence(token1[0].tolist(), vocab))

        
        seq_probs3 = self.forward_(token1, enc_output, category, signal=2)
        '''
        bsz, seq_len = token1.shape
        seq_probs3 = self.forward_(
                torch.stack([token1, fused_input], dim=1).view(-1, seq_len), 
                enlarge(enc_output, 2), 
                enlarge(category, 2), 
                signal=2
            )
        '''

        #tqdm.write(" Output 3: " + to_sentence(self.step(seq_probs3, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        #tqdm.write("--------------------------------------")
        outputs = ([seq_probs1, seq_probs2, seq_probs3],)
        return outputs

    def fuse_pred_gt(self, pred, gt, pad_mask):
        possibility = torch.rand(*gt.shape)
        possibility[pad_mask] = 1.0
        select_pred_ind = (possibility < self.select_out_ratio)
        input_ = gt.clone()
        input_[select_pred_ind] = pred[select_pred_ind]
        return input_

    def select_worst(self, token_probs, pad_mask):
        bsz, seq_len = token_probs.size()
        device = token_probs.device
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        '''
        possibility = torch.rand(bsz).to(device)
        ind = (possibility > 2*(self.masking_ratio - self.minimun_masking_ratio))
        possibility += 2*self.minimun_masking_ratio
        possibility[ind] = 2*self.masking_ratio
        ratio = possibility / 2
        '''

        
        ratio = torch.rand(bsz).to(device)
        
        num_mask = (seq_lens.float() * ratio).long()

        masks = torch.zeros(*token_probs.shape, device=token_probs.device)
        for i in range(masks.size(0)):
            ind = token_probs[i, :].topk(max(1, num_mask[i]), largest=False, sorted=False)[1]
            masks[i, ind] = 1
        return masks.byte()

    def select_random(self, token_probs, pad_mask):
        bsz, seq_len = token_probs.size()
        possibility = torch.rand(bsz, seq_len).to(token_probs.device)
        possibility[pad_mask] = 1.0

        '''
        tmp = torch.rand(bsz).to(token_probs.device)
        ind = (tmp > 2*(self.masking_ratio - self.minimun_masking_ratio))
        tmp += 2*self.minimun_masking_ratio
        tmp[ind] = 2*self.masking_ratio
        ratio = tmp / 2
        ratio = ratio.unsqueeze(1).repeat(1, seq_len)
        '''
        ratio = torch.rand(bsz).to(token_probs.device).unsqueeze(1).repeat(1, seq_len)
        return (possibility < ratio).byte()

class Signal3Decoder_alter(nn.Module):
    def __init__(self, config):
        super(Signal3Decoder_alter, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.bert = BertDecoder(config)
        self.embedding = self.bert.embedding
        self.signal_embedding = None

        self.masking_ratio = config.masking_ratio
        self.minimun_masking_ratio = 0.2
        self.select_out_ratio = config.select_out_ratio

        self.visual_tag = config.visual_tag
        self.nonvisual_tag = config.nonvisual_tag
        self.revision_tag = config.revision_tag

    def forward_(self, tgt_seq, enc_output, category, signal=0):
        bsz, seq_len = tgt_seq.shape
        assert signal in [0, 1, 2]

        signals = None

        seq_probs, *_ = self.bert(tgt_seq, enc_output, category, signals=signals)
        seq_probs = seq_probs[0]
        return seq_probs

    def step(self, seq_probs, pad_mask, tgt_word_prj, tag_replace=None):
        out = tgt_word_prj(seq_probs)
        probs = F.softmax(out, dim=-1) #[batch_size, max_len, vocab_size]
        token_probs, token = probs.max(dim=-1)
        token[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0

        if tag_replace is not None:
            source, target = tag_replace
            ind = token.eq(source)
            token[ind] = target
            copy_ = token_probs.clone()
            token_probs[ind] = 0.0
            return token, token_probs, copy_
        return token, token_probs

    def forward(self, tgt_seq, enc_output, category, tgt_word_prj, vocab):
        assert isinstance(tgt_seq, list)
        assert len(tgt_seq) == 3
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]

        pad_mask = tgt_seq[0].eq(Constants.PAD)

        seq_probs1 = self.forward_(tgt_seq[0], enc_output, category, signal=0)
        #tqdm.write(" Input  1: " + to_sentence(tgt_seq[0][0].tolist(), vocab))
        #tqdm.write(" Output 1: " + to_sentence(self.step(seq_probs1, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        seq_probs2 = self.forward_(tgt_seq[1], enc_output, category, signal=1)
        #tqdm.write(" Input  2: " + to_sentence(tgt_seq[1][0].tolist(), vocab))
        #tqdm.write(" Output 2: " + to_sentence(self.step(seq_probs2, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        
        #tqdm.write(" Input  3: " + to_sentence(tgt_seq[2][0].tolist(), vocab))
        seq_probs3 = self.forward_(tgt_seq[2], enc_output, category, signal=2)

        outputs = ([seq_probs1, seq_probs2, seq_probs3],)
        return outputs

    def fuse_pred_gt(self, pred, gt, pad_mask):
        possibility = torch.rand(*gt.shape)
        possibility[pad_mask] = 1.0
        select_pred_ind = (possibility < self.select_out_ratio)
        input_ = gt.clone()
        input_[select_pred_ind] = pred[select_pred_ind]
        return input_

    def select_worst(self, token_probs, pad_mask):
        bsz, seq_len = token_probs.size()
        device = token_probs.device
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        '''
        possibility = torch.rand(bsz).to(device)
        ind = (possibility > 2*(self.masking_ratio - self.minimun_masking_ratio))
        possibility += 2*self.minimun_masking_ratio
        possibility[ind] = 2*self.masking_ratio
        ratio = possibility / 2
        '''

        
        ratio = torch.rand(bsz).to(device)
        
        num_mask = (seq_lens.float() * ratio).long()

        masks = torch.zeros(*token_probs.shape, device=token_probs.device)
        for i in range(masks.size(0)):
            ind = token_probs[i, :].topk(max(1, num_mask[i]), largest=False, sorted=False)[1]
            masks[i, ind] = 1
        return masks.byte()

    def select_random(self, token_probs, pad_mask):
        bsz, seq_len = token_probs.size()
        possibility = torch.rand(bsz, seq_len).to(token_probs.device)
        possibility[pad_mask] = 1.0

        '''
        tmp = torch.rand(bsz).to(token_probs.device)
        ind = (tmp > 2*(self.masking_ratio - self.minimun_masking_ratio))
        tmp += 2*self.minimun_masking_ratio
        tmp[ind] = 2*self.masking_ratio
        ratio = tmp / 2
        ratio = ratio.unsqueeze(1).repeat(1, seq_len)
        '''
        ratio = torch.rand(bsz).to(token_probs.device).unsqueeze(1).repeat(1, seq_len)
        return (possibility < ratio).byte()

class Signal3Decoder(nn.Module):
    def __init__(self, config):
        super(Signal3Decoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.bert = BertDecoder(config)
        self.embedding = self.bert.embedding
        self.signal_embedding = nn.Embedding(2, config.dim_hidden) if not config.no_signal else None

        self.masking_ratio = config.masking_ratio
        self.minimun_masking_ratio = 0.2
        self.select_out_ratio = config.select_out_ratio

        self.visual_tag = config.visual_tag
        self.nonvisual_tag = config.nonvisual_tag
        self.revision_tag = config.revision_tag

    def forward_(self, tgt_seq, enc_output, category, signal):
        bsz, seq_len = tgt_seq.shape
        assert signal in [0, 1]

        if self.signal_embedding is None:
            signals = None
        else:
            signals = tgt_seq.new(bsz, seq_len).fill_(signal).long() # [batch_size, seq_len]
            signals = self.signal_embedding(signals)

        seq_probs, *_ = self.bert(tgt_seq, enc_output, category, signals=signals)
        seq_probs = seq_probs[0]
        return seq_probs

    def step(self, seq_probs, pad_mask, tgt_word_prj, tag_replace=None):
        out = tgt_word_prj(seq_probs)
        probs = F.softmax(out, dim=-1) #[batch_size, max_len, vocab_size]
        token_probs, token = probs.max(dim=-1)
        token[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0

        if tag_replace is not None:
            source, target = tag_replace
            ind = token.eq(source)
            token[ind] = target
            copy_ = token_probs.clone()
            token_probs[ind] = 0.0
            return token, token_probs, copy_
        return token, token_probs

    def forward(self, tgt_seq, enc_output, category, tgt_word_prj, vocab):
        assert isinstance(tgt_seq, list)
        assert len(tgt_seq) == 3
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]

        pad_mask = tgt_seq[0].eq(Constants.PAD)

        seq_probs1 = self.forward_(tgt_seq[0], enc_output, category, signal=0)
        #tqdm.write(" Input  1: " + to_sentence(tgt_seq[0][0].tolist(), vocab))
        #tqdm.write(" Output 1: " + to_sentence(self.step(seq_probs1, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        seq_probs2 = self.forward_(tgt_seq[1], enc_output, category, signal=0)
        #tqdm.write(" Input  2: " + to_sentence(tgt_seq[1][0].tolist(), vocab))
        #tqdm.write(" Output 2: " + to_sentence(self.step(seq_probs2, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        
        token1, token_probs1, copy_1 = self.step(seq_probs1, pad_mask, tgt_word_prj, tag_replace=[self.visual_tag, self.revision_tag])
        token2, token_probs2, copy_2 = self.step(seq_probs2, pad_mask, tgt_word_prj, tag_replace=[self.nonvisual_tag, self.revision_tag])

        ind_blank = token1.eq(self.visual_tag) & token2.eq(self.nonvisual_tag)
        ind = token_probs2 > token_probs1
        token1[ind] = token2[ind]
        #tqdm.write("  Fusion : " + to_sentence(token1[0].tolist(), vocab))
        

        mask_ind = token1.eq(self.revision_tag) | tgt_seq[2].eq(self.revision_tag)
        #tqdm.write(" Input 31: " + to_sentence(tgt_seq[2][0].tolist(), vocab))
        if self.select_out_ratio:
            possibility = torch.rand(*tgt_seq[2].shape)
            possibility[pad_mask] = 1.0
            possibility[mask_ind] = 1.0
            select_out_ind = (possibility < self.select_out_ratio)
            input_ = tgt_seq[2].clone()
            input_[select_out_ind] = token1[select_out_ind]
            #tqdm.write(" Input 22: " + to_sentence(input_[0].tolist(), vocab))
        else:
            input_ = tgt_seq[2]

        #tqdm.write(" Input  32: " + to_sentence(input_[0].tolist(), vocab))

        
        seq_probs3 = self.forward_(input_, enc_output, category, signal=1)
        outputs = ([seq_probs1, seq_probs2, seq_probs3],)
        return outputs

class NVDecoder(nn.Module):
    def __init__(self, config):
        super(NVDecoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.bert = BertDecoder(config)
        self.embedding = self.bert.embedding
        self.signal_embedding = nn.Embedding(2, config.dim_hidden) if not config.no_signal else None
        #self.signal_embedding = None

        self.masking_ratio = config.masking_ratio
        self.minimun_masking_ratio = 0.2
        self.select_out_ratio = config.select_out_ratio

        self.visual_tag = config.visual_tag
        self.nonvisual_tag = config.nonvisual_tag
        self.revision_tag = config.revision_tag

    def forward_(self, tgt_seq, enc_output, category, signal, tags=None):
        bsz, seq_len = tgt_seq.shape
        assert signal in [0, 1]

        if self.signal_embedding is None:
            signals = None
        else:
            signals = tgt_seq.new(bsz, seq_len).fill_(signal).long() # [batch_size, seq_len]
            signals = self.signal_embedding(signals)

        seq_probs, embs, *_ = self.bert(tgt_seq, enc_output, category, signals=signals, tags=tags)
        seq_probs = seq_probs[0]
        return seq_probs, embs

    def step(self, seq_probs, pad_mask, tgt_word_prj, tag_replace=None):
        out = tgt_word_prj(seq_probs)
        probs = F.softmax(out, dim=-1) #[batch_size, max_len, vocab_size]
        token_probs, token = probs.max(dim=-1)
        token[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0

        if tag_replace is not None:
            source, target = tag_replace
            ind = token.eq(source)
            token[ind] = target
            copy_ = token_probs.clone()
            token_probs[ind] = 0.0
            return token, token_probs, copy_
        return token, token_probs

    def forward(self, tgt_seq, enc_output, category, tgt_word_prj, vocab, tags=None):
        assert isinstance(tgt_seq, list)
        assert len(tgt_seq) == 2
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]

        pad_mask = tgt_seq[0].eq(Constants.PAD)

        seq_probs1, _ = self.forward_(tgt_seq[0], enc_output, category, signal=0, tags=tags)
        #tqdm.write(" Input  1: " + to_sentence(tgt_seq[0][0].tolist(), vocab))
        #tqdm.write(" Output 1: " + to_sentence(self.step(seq_probs1, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        token1, token_probs1 = self.step(seq_probs1, pad_mask, tgt_word_prj)

        mask_ind = token1.eq(self.revision_tag) | tgt_seq[1].eq(self.revision_tag)
        #tqdm.write(" Input 21: " + to_sentence(tgt_seq[1][0].tolist(), vocab))
        if self.select_out_ratio:
            possibility = torch.rand(*tgt_seq[1].shape)
            possibility[pad_mask] = 1.0
            possibility[mask_ind] = 1.0
            select_out_ind = (possibility < self.select_out_ratio)
            input_ = tgt_seq[1].clone()
            input_[select_out_ind] = token1[select_out_ind]
            #tqdm.write(" Input 22: " + to_sentence(input_[0].tolist(), vocab))
        else:
            input_ = tgt_seq[1]
        #tqdm.write(" Input 22: " + to_sentence(input_[0].tolist(), vocab))
        seq_probs2, embs = self.forward_(input_, enc_output, category, signal=1, tags=tags)
        #tqdm.write(" Output 2: " + to_sentence(self.step(seq_probs2, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        
        outputs = ([seq_probs1, seq_probs2],embs,)
        return outputs

import copy
class VIP_layer(nn.Module):
    def __init__(self, seq_len, VIP_level=3):
        super(VIP_layer, self).__init__()
        
        VIP_level_limit = math.log(seq_len, 2)
        assert int(VIP_level_limit) == VIP_level_limit, 'seq_len must be a exponent of 2'
        assert VIP_level <= VIP_level_limit + 1, \
        'The maximun VIP_level for seq_len({}) is {:d}, {} is not allowed'.format(seq_len, VIP_level_limit+1, VIP_level)

        kernel_size = [(seq_len//(2**n), 1, 1) for n in range(VIP_level)]
        dilation = [(2**n, 1, 1) for n in range(VIP_level)]

        # Specific-timescale Pooling
        self.pooling = nn.ModuleList(
                [copy.deepcopy(
                    nn.MaxPool3d(
                        kernel_size=ks,
                        stride=1,
                        padding=0,
                        dilation=di
                    )
                ) for ks, di in zip(kernel_size, dilation)]
            )

        self.VIP_level = VIP_level


    def forward(self, x):
        '''
            input: 
                -- x: [batch_size, seq_len, dim_feats]
            output:
                -- the prediction results: [batch_size, num_class]
        '''   
        bsz, _, dim = x.shape
        tmp = x.clone()
        x = x.permute(0,2,1).unsqueeze(3).unsqueeze(4) #[batch_size, dim_feats, D=seq_len, H=1, W=1]  
        collections = []

        for n in range(self.VIP_level):
            y = self.pooling[n](x).mean(2) #[batch_size, dim_feats, H=1, W=1]
            collections.append(y.contiguous().view(bsz, dim))

        return torch.stack(collections, dim=1).view(bsz*self.VIP_level, dim)


class MSDecoder(nn.Module):
    def __init__(self, config):
        super(MSDecoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.bert = BertDecoder(config)
        self.embedding = self.bert.embedding
        self.masking_ratio = config.masking_ratio
        self.minimun_masking_ratio = 0.2
        self.select_out_ratio = config.select_out_ratio

        self.visual_tag = config.visual_tag
        self.nonvisual_tag = config.nonvisual_tag
        self.revision_tag = config.revision_tag

        self.multiscale = config.multiscale
        #self.vip = VIP_layer(config.n_frames*2, VIP_level=self.multiscale)

    def forward_(self, tgt_seq, enc_output, category, multiscale):
        bsz, seq_len = tgt_seq.shape

        if multiscale:
            #signals = self.vip(enc_output)
            #signals = signals.unsqueeze(1).repeat(1, tgt_seq.size(1), 1)
            image, motion = enc_output.chunk(2, dim=1)
            signals = torch.stack([
                image.mean(1).unsqueeze(1).repeat(1, tgt_seq.size(1), 1), 
                motion.mean(1).unsqueeze(1).repeat(1, tgt_seq.size(1), 1)
                ], dim=1).view(bsz*2, tgt_seq.size(1), -1)

            seq_probs, *_ = self.bert(enlarge(tgt_seq, self.multiscale), enlarge(enc_output, self.multiscale), enlarge(category, self.multiscale), signals=signals)
        else:
            signals = None
            seq_probs, *_ = self.bert(tgt_seq, enc_output, category, signals=signals)
        seq_probs = seq_probs[0]
        return seq_probs

    def step(self, seq_probs, pad_mask, tgt_word_prj, tag_replace=None):
        out = tgt_word_prj(seq_probs)
        probs = F.softmax(out, dim=-1) #[batch_size, max_len, vocab_size]
        token_probs, token = probs.max(dim=-1)
        token[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0

        if tag_replace is not None:
            source, target = tag_replace
            ind = token.eq(source)
            token[ind] = target
            copy_ = token_probs.clone()
            token_probs[ind] = 0.0
            return token, token_probs, copy_
        return token, token_probs

    def forward(self, tgt_seq, enc_output, category, tgt_word_prj, vocab):
        assert isinstance(tgt_seq, list)
        assert len(tgt_seq) == 2
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]

        pad_mask = tgt_seq[0].eq(Constants.PAD)

        seq_probs1 = self.forward_(tgt_seq[0], enc_output, category, multiscale=True)
        #tqdm.write(" Input  1: " + to_sentence(tgt_seq[0][0].tolist(), vocab))
        #tqdm.write(" Output 1: " + to_sentence(self.step(seq_probs1, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))

        seq_probs2 = self.forward_(tgt_seq[1], enc_output, category, multiscale=False)
        #tqdm.write(" Output 2: " + to_sentence(self.step(seq_probs2, pad_mask, tgt_word_prj)[0][0].tolist(), vocab))
        
        outputs = ([seq_probs1, seq_probs2],)
        return outputs


'''
class Attribute_Predictor_Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, dim_hidden):
        super(Attribute_Predictor_Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim_hidden, padding_idx=Constants.PAD)
        self.position_embeddings = nn.Embedding(config.max_len, dim_hidden)
        self.category_embeddings = nn.Embedding(config.num_category, dim_hidden) if config.with_category else None
        self.return_pos = return_pos

        self.LN_lang = BertLayerNorm(dim_hidden, eps=config.layer_norm_eps)
        self.LN_ap = BertLayerNorm(dim_hidden, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.tgt_word_prj = nn.Linear(dim_hidden, config.vocab_size, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        self.tag2hidden = None

    def forward(self, input_ids, category=None, position_ids=None, additional_feats=None, tags=None, task=0):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        
        if task == 0:
            words_embeddings = self.word_embeddings(input_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = position_embeddings

        if self.category_embeddings is not None:
            assert category is not None
            category_embeddings = self.category_embeddings(category).repeat(1, seq_length, 1)
            embeddings += category_embeddings

        if additional_feats is not None:
            embeddings += additional_feats

        if self.tag2hidden is not None:
            assert tags is not None
            embeddings += self.tag2hidden(tags).unsqueeze(1).repeat(1, seq_length, 1)

        embeddings = self.LN_lang(embeddings) if task == 0 else self.LN_ap(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def linear(self, x):
        return self.tgt_word_prj(x)
'''



class BeamDecoder(nn.Module):
    def __init__(self, config, embedding):
        super(BeamDecoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.output_attentions = config.output_attentions
        #self.embedding = BertEmbeddings(config, return_pos=True if config.pos_attention else False)
        #self._init_embedding(embedding.word_embeddings.weight)

        self.embedding = embedding
        self.bd_load_feats = getattr(config, "bd_load_feats", False)
        self.layer = nn.ModuleList([BertLayer(config, decoder_layer=True if self.bd_load_feats else False) for _ in range(config.num_hidden_layers_decoder)])

        self.prj = nn.Sequential(
                nn.Linear(config.dim_hidden, config.dim_hidden),
                nn.ReLU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.dim_hidden, 2, bias=False),
            )


    def _init_embedding(self, weight, option={}):
        self.embedding.word_embeddings.weight.data.copy_(weight.data)
        if not option.get('train_emb', True):
            for p in self.embedding.word_embeddings.parameters():
                p.requires_grad = False

    def forward(self, enc_output, tgt_seq, category):
        if self.bd_load_feats:
            if isinstance(enc_output, list):
                enc_output = enc_output[0]
            src_seq = torch.ones(enc_output.size(0), enc_output.size(1)).to(enc_output.device)
            attend_to_enc_output_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)
        else:
            enc_output = attend_to_enc_output_mask = None
        all_attentions = ()

        # each token can attend to the whole sequence
        slf_attn_mask = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask[:, :, 0] = 1

        non_pad_mask = get_non_pad_mask(tgt_seq)
        hidden_states = self.embedding(tgt_seq, category=category)

        for i, layer_module in enumerate(self.layer):
            if not i:
                input_ = hidden_states
            else:
                input_ = layer_outputs[0]# + hidden_states
            layer_outputs = layer_module(
                    hidden_states=input_, 
                    enc_output=enc_output,
                    attention_mask=slf_attn_mask, 
                    non_pad_mask=non_pad_mask,
                    attend_to_enc_output_mask=attend_to_enc_output_mask
                )

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)

        logit = self.prj(layer_outputs[0][:, 0]) # only feed the [cls] token into the classifier
        logit = F.log_softmax(logit, dim=-1)

        outputs = (logit,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
