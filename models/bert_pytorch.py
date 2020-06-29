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

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, return_pos=False):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim_hidden, padding_idx=Constants.PAD)
        self.position_embeddings = nn.Embedding(config.max_len, config.dim_hidden)
        self.category_embeddings = nn.Embedding(config.num_category, config.dim_hidden) if config.with_category else None
        self.return_pos = return_pos

        self.LayerNorm = BertLayerNorm(config.dim_hidden, eps=config.layer_norm_eps)
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
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings
        else:
            embeddings = words_embeddings
            if self.category_embeddings is not None:
                embeddings += category_embeddings
            embeddings = self.dropout(self.LayerNorm(embeddings))
            position_embeddings = self.dropout(self.LayerNorm(position_embeddings))

            return embeddings, position_embeddings
        #return words_embeddings, position_embeddings
            


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
        self.dense = nn.Linear(config.dim_hidden, config.dim_hidden)
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
        outputs = self.dense(outputs)

        return (outputs, attention_probs) if self.output_attentions else (outputs,)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, config):
        super(SublayerConnection, self).__init__()
        self.norm = BertLayerNorm(config.dim_hidden)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        output = sublayer(self.norm(x))
        if isinstance(output, tuple):
            output, *other = output
        else:
            other = None
        return x + self.dropout(output), other


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(config.dim_hidden, config.intermediate_size)
        self.w_2 = nn.Linear(config.intermediate_size, config.dim_hidden)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = ACT2FN[config.hidden_act]
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class BertLayer(nn.Module):
    def __init__(self, config, decoder_layer=True):
        super(BertLayer, self).__init__()
        self.attend_to_word_itself = BertSelfAttention(config)
        self.sublayer_word = SublayerConnection(config)

        if config.pos_attention:
            self.attend_to_position = BertSelfAttention(config)
            self.sublayer_pos = SublayerConnection(config)
        else:
            self.attend_to_position, self.sublayer_pos = None, None

        if decoder_layer:
            self.attend_to_enc_output = BertSelfAttention(config)
            self.sublayer_eo = SublayerConnection(config)
        else:
            self.attend_to_enc_output, self.sublayer_eo = None, None
        
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer_output = SublayerConnection(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, enc_output=None, non_pad_mask=None, attention_mask=None, attend_to_enc_output_mask=None, position_embeddings=None):
        others = ()
        hidden_states, other = self.sublayer_word(hidden_states, lambda _x: self.attend_to_word_itself(_x, _x, _x, attention_mask))
        #print(type(others), type(other))
        others = others + tuple(other)
        if non_pad_mask is not None: hidden_states *= non_pad_mask

        if self.attend_to_position is not None:
            hidden_states, other = self.sublayer_pos(hidden_states, lambda _x: self.attend_to_position(position_embeddings, position_embeddings, _x, attention_mask))
            others = others + tuple(other)
            if non_pad_mask is not None: hidden_states *= non_pad_mask

        if self.attend_to_enc_output is not None:
            hidden_states, other = self.sublayer_eo(hidden_states, lambda _x: self.attend_to_enc_output(_x, enc_output, enc_output, attend_to_enc_output_mask))
            others = others + tuple(other)
            if non_pad_mask is not None: hidden_states *= non_pad_mask

        hidden_states, _ = self.sublayer_output(hidden_states, self.feed_forward)
        hidden_states = self.dropout(hidden_states)
        return (hidden_states,) + others


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
    def __init__(self, config=BertConfig()):
        super(BertDecoder, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.emb = BertEmbeddings(config, return_pos=True if config.pos_attention else False)
        self.layer = nn.ModuleList([BertLayer(config, decoder_layer=True) for _ in range(config.num_hidden_layers_decoder)])
        self.pos_attention = config.pos_attention
        self.enhance_input = config.enhance_input
        self.watch = config.watch
        self.decoder_type = config.decoder_type

    def _init_embedding(self, weight, option={}):
        self.emb.word_embeddings.weight.data.copy_(weight.data)
        if not option.get('train_emb', True):
            for p in self.emb.word_embeddings.parameters():
                p.requires_grad = False

    def forward(self, tgt_seq, enc_output, category):
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]
        all_attentions = ()

        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        if self.decoder_type == 'NARFormer':
            slf_attn_mask = slf_attn_mask_keypad
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

        if self.pos_attention:
            hidden_states, position_embeddings = self.emb(tgt_seq, category=category)
        else:
            hidden_states = self.emb(tgt_seq, additional_feats=additional_feats, category=category)
            position_embeddings = None

        res = []
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, enc_output=enc_output, 
                attention_mask=slf_attn_mask, attend_to_enc_output_mask=attend_to_enc_output_mask, position_embeddings=position_embeddings, non_pad_mask=non_pad_mask)
            hidden_states = layer_outputs[0]
            res.append(hidden_states)
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        res = [res[-1]]
        outputs = (res,)

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
