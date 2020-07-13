from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
'''
class Attentional_GRU(nn.Module):
    def __init__(self, dim_input, dim_hidden, opt):
        super(Attentional_GRU, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden

        self.query = nn.Linear(dim_input, dim_hidden)
        self.query_ln = nn.LayerNorm(dim_hidden)
        self.emb = None
        self.input_type = opt.get('AGRU_input_type', 'concat')
        if self.input_type == 'concat':
            self.rnn = nn.GRUCell(dim_input + dim_hidden, dim_hidden)
        else:
            self.rnn = nn.GRUCell(dim_hidden, dim_hidden)
            self.input_ln = nn.LayerNorm(dim_hidden)
        self.mha = MultiHeadAttention(dim_hidden, head=1)
        #self.mha = BasicAttention(dim_hidden, [dim_hidden], 512)
        self.dropout = nn.Dropout(0.5)

    def forward(self, feats, pre_outputs, hx):
        if len(hx.shape) == 3:
            hx = hx.squeeze(0)
        bsz, seq_len, _ = feats.shape
        outputs = []
        attention_probs = []

        for i in range(seq_len):
            query = self.query_ln(self.query(feats[:, i, :]))
            context, att = self.mha(query.unsqueeze(1), pre_outputs, pre_outputs)
            if self.input_type == 'concat':
                input_ = torch.cat([context.squeeze(1), feats[:, i, :]], dim=1)
            else:
                input_ = self.input_ln(context.squeeze(1) + feats[:, i, :])
            #context, att = self.mha(hx, [pre_outputs])
            #input_ = torch.cat([context, feats[:, i, :]], dim=1)
            hx = self.rnn(self.dropout(input_), hx)
            outputs.append(hx.clone())
            attention_probs.append(att)

        return torch.stack(outputs, dim=1), hx, torch.cat(attention_probs, dim=1)

class Hierarchical_Encoder(nn.Module):
    def __init__(self, input_size=[1536, 2048], hidden_size=[512, 1024], opt=None):
        super(Hierarchical_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_dropout_p = opt['encoder_dropout']
        self.dropout = nn.Dropout(self.input_dropout_p)

        self.rnn_list = []
        for i in range(len(input_size)):
            if not i:
                tmpRNN = nn.GRU(self.input_size[i], self.hidden_size[i], batch_first=True)
            else:
                tmpRNN = Attentional_GRU(self.input_size[i], self.hidden_size[i], opt)
            self.add_module("rnn%d"%i, tmpRNN)
            self.rnn_list.append(tmpRNN)

    def forward(self, input_feats):
        assert len(input_feats) == len(self.input_size)
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

        encoder_hidden = None
        for i in range(len(input_feats)):
            if i:
                rnn_output, encoder_hidden, attention_probs = self.rnn_list[i](input_feats[i], rnn_output, encoder_hidden)
            else:
                rnn_output, encoder_hidden = self.rnn_list[i](self.dropout(input_feats[i]), encoder_hidden)

        return rnn_output, encoder_hidden


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_hidden, head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.head = head
        self.d_k = dim_hidden // head
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attention_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_k, self.head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if attention_mask is not None:
            attention_mask = attention_mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        attention_scores = torch.bmm(q, k.transpose(1, 2))
        attention_scores = attention_scores / math.sqrt(self.d_k)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        #print("after att:", attention_probs)
        attention_probs = self.dropout(attention_probs)

        outputs = torch.bmm(attention_probs, v)

        outputs = outputs.view(n_head, sz_b, len_q, d_v)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        return outputs, attention_probs

'''

class HighWay(nn.Module):
    def __init__(self, dim_hidden):
        super(HighWay, self).__init__()

        self.w1 = nn.Linear(dim_hidden, dim_hidden)
        self.w2 = nn.Linear(dim_hidden, dim_hidden)
        self.tanh = nn.Tanh()

    def forward(self, x):
        y = self.tanh(self.w1(x))
        gate = torch.sigmoid(self.w2(x))
        return gate * x + (1 - gate) * y


class Encoder_Baseline(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], encoder_type='gru', together=False):
        super(Encoder_Baseline, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.module_type = 0 if encoder_type == 'linear' else 1
        self.dropout = nn.Dropout(0.5)
        num_layers = 1
        if encoder_type == 'gru':
            tmp_encoder = nn.GRU
        elif 'lstm' in encoder_type:
            tmp_encoder = nn.LSTM
        else:
            tmp_encoder = nn.Linear

        if encoder_type == 'mslstm':
            num_layers = 2

        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        self.encoder = []
        self.together = together
        if together:
            if self.module_type:
                tmp_module = tmp_encoder(sum(input_size), output_size[0], batch_first=True, num_layers=num_layers)
            else:
                tmp_module = tmp_encoder(sum(input_size), output_size[0])
            self.add_module("Encoder", tmp_module)
            self.encoder.append(tmp_module)
        else:
            for i in range(len(self.input_size)):
                if input_size[i] is None:
                    self.encoder.append(None)
                else:
                    tmp_module = tmp_encoder(input_size[i], output_size[i], batch_first=True, num_layers=num_layers) if self.module_type else \
                                 tmp_encoder(input_size[i], output_size[i])
                    '''
                    tmp_module = nn.Sequential(
                        *(
                            nn.Linear(input_size[i], output_size[i]),
                            HighWay(output_size[i]),
                            #nn.Dropout(0.5),
                        )
                    )
                    '''
                    self.add_module("Encoder%s"%(name[i]), tmp_module)
                    self.encoder.append(tmp_module)

        self._init_weights()

    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Linear):
                print('initial baseline encoder weights')
                torch.nn.init.xavier_normal_(module.weight)

    def forward(self, input_feats):
        assert len(input_feats) == len(self.input_size)
        if not len(input_feats): return [], []
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

        outputs = []
        hiddens = []
        if self.together:
            if self.module_type:
                eo, eh = self.encoder[0](self.dropout(torch.cat(input_feats, dim=2)))
            else:
                #eo = F.relu(self.encoder[i](self.dropout(input_feats[i])))
                eo = self.encoder[0](self.dropout(torch.cat(input_feats, dim=2)))
                eh = eo.mean(1)
            outputs.append(eo)
            hiddens.append(eh)
        else:
            for i in range(len(input_feats)):
                if self.encoder[i] is None:
                    eo = input_feats[i]
                    eh = eo.mean(1)
                else:
                    if self.module_type:
                        eo, eh = self.encoder[i](self.dropout(input_feats[i]))
                        if isinstance(eh, tuple):
                            eh = (eh[0][-1, :, :], eh[1][-1, :, :])
                        else:
                            eh = eh[-1, :, :]
                    else:
                        #eo = F.relu(self.encoder[i](self.dropout(input_feats[i])))
                        #eo = F.tanh(self.encoder[i](self.dropout(input_feats[i])))
                        eo = self.encoder[i](self.dropout(input_feats[i]))
                        eh = eo.mean(1)
                outputs.append(eo)
                hiddens.append(eh)

        return outputs, hiddens


'''
class Encoder_Baseline(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], encoder_type='gru', together=False):
        super(Encoder_Baseline, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        self.encoder = []
        self.together = together
        for i in range(len(self.input_size)):
            if input_size[i] is None:
                self.encoder.append(None)
            else:
                tmp_module = nn.Sequential(
                    nn.Linear(input_size[i], output_size[i]),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(output_size[i], output_size[i]),
                    nn.Tanh()
                )
                self.add_module("Encoder%s"%(name[i]), tmp_module)
                self.encoder.append(tmp_module)

        self._init_weights()

    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Linear):
                print('initial baseline encoder weights')
                torch.nn.init.xavier_normal_(module.weight)

    def forward(self, input_feats):
        assert len(input_feats) == len(self.input_size)
        if not len(input_feats): return [], []
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

        outputs = []
        hiddens = []
        for i in range(len(input_feats)):
            if self.encoder[i] is None:
                eo = input_feats[i]
                eh = eo.mean(1)
            else:
                eo = self.encoder[i](input_feats[i])
                eh = eo.mean(1)
            outputs.append(eo)
            hiddens.append(eh)

        return outputs, hiddens
'''
class Hierarchical_Encoder(nn.Module):
    def __init__(self, input_size=[1536, 2048], hidden_size=[512, 1024], opt=None):
        super(Hierarchical_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_dropout_p = opt['encoder_dropout']
        self.dropout = nn.Dropout(self.input_dropout_p)

        rnn = nn.GRU
        self.no_global_context = opt.get('no_global_context', False)
        self.no_regional_context = opt.get('no_regional_context', False)
        print('Global {}\tRegional {}'.format(not self.no_global_context, not self.no_regional_context))
        for i in range(len(input_size)):
            if i and not self.no_regional_context:
                input_size = self.hidden_size[i-1] + self.input_size[i]
            else:
                input_size = self.input_size[i]

            tmpRNN = rnn(input_size, self.hidden_size[i], batch_first=True, num_layers=1)
            self.add_module("rnn%d"%i, tmpRNN)
        self.rnn_list = []
        for name, module in self.named_children():
            if 'rnn' in name: self.rnn_list.append(module)



    def forward(self, input_feats):
        assert len(input_feats) == len(self.input_size)
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len


        encoder_hidden = None
        for i in range(len(input_feats)):
            if i:
                if self.no_global_context:
                    encoder_hidden = None
                if self.no_regional_context:
                    input_ = input_feats[i]
                else:
                    input_ = torch.cat([rnn_output, input_feats[i]], dim=2)
                rnn_output, encoder_hidden = self.rnn_list[i](self.dropout(input_), encoder_hidden)
            else:
                rnn_output, encoder_hidden = self.rnn_list[i](self.dropout(input_feats[i]), encoder_hidden)

        return rnn_output, encoder_hidden[-1, :, :]

class LSTM_Decoder(nn.Module):
    def __init__(self, opt, embedding=None, num_modality=-1):
        super(LSTM_Decoder, self).__init__()
        with_multimodal_attention = opt.get('with_multimodal_attention', False)
        addition = opt.get('addition', False)
        gated_sum = opt.get('gated_sum', False)
        if num_modality == -1:
            num_modality = 1 if (opt['encoder_type'] == 'IPE' or addition or gated_sum) else (len(opt['modality']) - sum(opt['skip_info']))

        self.word_size = opt['dim_hidden']
        if opt['encoder_type'] == 'IPE' or with_multimodal_attention or addition or gated_sum:
            self.feats_size = opt['dim_hidden']  
        else:
            self.feats_size = opt['dim_hidden'] * num_modality

        self.hidden_size = opt["dim_hidden"]
        self.vocab_size = opt["vocab_size"]
        self.max_len = opt['max_len']

        self.embedding = embedding if embedding is not None else nn.Embedding(self.vocab_size, self.word_size)


        lstm_input_size = self.word_size + self.feats_size + (opt['num_category'] if opt['with_category'] else 0)
        

        self.with_category = opt['with_category']
        self.rnn = nn.LSTMCell(lstm_input_size, self.hidden_size)
        self.forget_bias = opt.get('forget_bias', 0.6)
        self._init_lstm_forget_bias()
        
        '''
        self.att = BasicAttention(
                self.hidden_size, 
                [self.feats_size], 
                opt.get('att_mid_size', 256),
            )
        '''
        '''
        if opt['encoder_type'] == 'GRU':
            if opt['multi_scale_context_attention'] and not opt.get('query_all', False):
                length = 1
            else:
                length = len(opt['modality'])
        else:
            length = 1
        '''
        

        self.att = Attentional_Attention(
                self.hidden_size, 
                [self.hidden_size] * num_modality, 
                opt.get('att_mid_size', 256),
                with_multimodal_attention=with_multimodal_attention
            )
        self.dropout = nn.Dropout(opt['decoder_dropout'])

    def _init_lstm_forget_bias(self):
        print("====> forget bias %g" % (self.forget_bias))
        for name, module in self.named_children():
            if 'rnn' in name: 
                ingate, forgetgate, cellgate, outgate = module.bias_ih.chunk(4, 0)
                forgetgate += self.forget_bias / 2
                module.bias_ih = Parameter(torch.cat([ingate, forgetgate, cellgate, outgate], dim=0))

                ingate, forgetgate, cellgate, outgate = module.bias_hh.chunk(4, 0)
                forgetgate += self.forget_bias / 2
                module.bias_hh = Parameter(torch.cat([ingate, forgetgate, cellgate, outgate], dim=0))
        

    def init_hidden(self, encoder_hidden):
        if isinstance(encoder_hidden, tuple):
            if len(encoder_hidden[0].shape) == 3:
                return (encoder_hidden[0].squeeze(0).clone(), encoder_hidden[1].squeeze(0).clone())
            else:
                return (encoder_hidden[0].clone(), encoder_hidden[1].clone())

        if len(encoder_hidden.shape) == 3:
            return (encoder_hidden.squeeze(0).clone(), encoder_hidden.squeeze(0).clone())
        else:
            return (encoder_hidden.clone(), encoder_hidden.clone())


    def forward(self, it, encoder_outputs, category, decoder_hidden):
        current_words = self.embedding(it)
        hidden_state = decoder_hidden[0] if isinstance(decoder_hidden, tuple) else decoder_hidden
        context, frames_weight = self.att(hidden_state, encoder_outputs if isinstance(encoder_outputs, list) else [encoder_outputs])

        if self.with_category:
            #print(current_words.dtype, context.dtype, category.dtype)
            #print(category.shape)
            input_content = torch.cat([current_words, context, category], dim=1)
        else:
            input_content = torch.cat([current_words, context], dim=1)

        input_content = self.dropout(input_content)
        decoder_hidden = self.rnn(input_content, decoder_hidden)

        return self.dropout(decoder_hidden[0]), decoder_hidden, frames_weight


class ENSEMBLE_Decoder(nn.Module):
    def __init__(self, opt, embedding=None):
        super(ENSEMBLE_Decoder, self).__init__()

        if not opt.get('no_shared_word_emb', False):
            vocab_size, word_size = opt["vocab_size"], opt['dim_hidden']
            self.embedding = nn.Embedding(vocab_size, word_size)
        else:
            self.embedding = None

        self.decoder = []
        self.num_decoder = (len(opt['modality']) - sum(opt['skip_info']))
        for i in range(self.num_decoder):
            tmp_decoder = LSTM_Decoder(opt, embedding=self.embedding, num_modality=1)
            self.add_module('lstm%d'%i, tmp_decoder)
            self.decoder.append(tmp_decoder)


    def init_hidden(self, encoder_hidden):
        assert isinstance(encoder_hidden, list)
        assert len(encoder_hidden) == self.num_decoder

        decoder_hidden = []
        for i in range(len(encoder_hidden)):
            decoder_hidden.append(self.decoder[i].init_hidden(encoder_hidden[i]))
        return decoder_hidden

    def forward(self, it, encoder_outputs, category, decoder_hidden):
        outputs = []
        hiddens = []
        frames_weight = []
        assert isinstance(decoder_hidden, list)
        assert len(decoder_hidden) == self.num_decoder

        for i in range(self.num_decoder):
            o, h, fw = self.decoder[i](it, encoder_outputs, category, decoder_hidden[i])
            outputs.append(o)
            hiddens.append(h)
            frames_weight.append(fw)

        return outputs, hiddens, frames_weight


class LSTM_Decoder2(nn.Module):
    def __init__(self, opt):
        super(LSTM_Decoder2, self).__init__()
        self.word_size = opt['dim_hidden']
        self.feats_size = opt['dim_hidden']
        self.hidden_size = opt["dim_hidden"]
        self.vocab_size = opt["vocab_size"]
        self.max_len = opt['max_len']

        self.embedding = nn.Embedding(self.vocab_size, self.word_size)

        self.with_category = opt['with_category']
        self.rnn = nn.LSTMCell(self.word_size + (opt['num_category'] if opt['with_category'] else 0), self.hidden_size)
        self.rnn2 = nn.LSTMCell(self.hidden_size + self.feats_size, self.hidden_size)
        self.forget_bias = opt.get('forget_bias', 0.6)
        self._init_lstm_forget_bias()
        
        '''
        self.att = BasicAttention(
                self.hidden_size, 
                [self.feats_size], 
                opt.get('att_mid_size', 256),
            )
        '''
        
        if opt['encoder_type'] == 'GRU':
            if opt['multi_scale_context_attention'] and not opt.get('query_all', False):
                length = 1
            else:
                length = len(opt['modality'])
        else:
            length = 1
        self.att = Attentional_Attention(
                self.hidden_size, 
                [self.feats_size] * length, 
                opt.get('att_mid_size', 256),

            )
        self.dropout = nn.Dropout(opt['decoder_dropout'])

    def _init_lstm_forget_bias(self):
        print("====> forget bias %g" % (self.forget_bias))
        for name, module in self.named_children():
            if 'rnn' in name: 
                ingate, forgetgate, cellgate, outgate = module.bias_ih.chunk(4, 0)
                forgetgate += self.forget_bias / 2
                module.bias_ih = Parameter(torch.cat([ingate, forgetgate, cellgate, outgate], dim=0))

                ingate, forgetgate, cellgate, outgate = module.bias_hh.chunk(4, 0)
                forgetgate += self.forget_bias / 2
                module.bias_hh = Parameter(torch.cat([ingate, forgetgate, cellgate, outgate], dim=0))
        

    def init_hidden(self, encoder_hidden):
        if len(encoder_hidden.shape) == 3:
            return (encoder_hidden.squeeze(0).clone(), encoder_hidden.squeeze(0).clone())
        else:
            return (encoder_hidden.clone(), encoder_hidden.clone())


    def forward(self, it, encoder_outputs, category, decoder_hidden):
        current_words = self.embedding(it)
        assert isinstance(decoder_hidden, list)
        assert len(decoder_hidden) == 2
        if self.with_category:
            input_content = torch.cat([current_words, category], dim=1)
        else:
            input_content = current_words

        decoder_hidden[0] = self.rnn(self.dropout(input_content), decoder_hidden[0])
        hidden_state = decoder_hidden[0][0]
        context, frames_weight = self.att(hidden_state, encoder_outputs if isinstance(encoder_outputs, list) else [encoder_outputs])
        input_content = torch.cat([hidden_state, context], dim=1)
        
        decoder_hidden[1] = self.rnn(self.dropout(input_content), decoder_hidden[1])

        return self.dropout(decoder_hidden[1][0]), decoder_hidden, frames_weight


class BasicAttention(nn.Module):
    def __init__(self, 
        dim_hidden, dim_feats, dim_mid, 
        activation=F.tanh, activation_type='acc', fusion_type='addition'):
        super(BasicAttention, self).__init__()
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

    def forward(self, hidden_state, feats, enhance_feats=None, category=None, t=None):
        """
        feats: [batch_size, seq_len, dim]
        """

        if len(hidden_state.shape) == 1:
            hidden_state = hidden_state.unsqueeze(0)
        if len(hidden_state.shape) == 3 and hidden_state.shape[0] == 1:
            hidden_state = hidden_state.squeeze(0)

        context = []
        all_weight = []
        for i in range(self.num_feats):
            _, weight = self.cal_out(
                [self.linear1_h, self.linear1_f[i]],
                self.linear2_temporal,
                [hidden_state, feats[i]]
                )
            all_weight.append(weight)
            context.append(torch.bmm(weight.unsqueeze(1), feats[i]).squeeze(1))

        final_context = torch.cat(context, dim=1)


        return final_context, torch.cat(all_weight, dim=1)


class Attentional_Attention(nn.Module):
    def __init__(self, 
        dim_hidden, dim_feats, dim_mid, 
        activation=F.tanh, activation_type='acc', fusion_type='addition', with_multimodal_attention=False):
        super(Attentional_Attention, self).__init__()
        assert isinstance(dim_feats, list)
        self.num_feats = len(dim_feats)
        self.dim_hidden = dim_hidden
        self.dim_feats = dim_feats
        self.dim_mid = dim_mid
        self.activation = activation
        self.activation_type = activation_type
        self.fusion_type = fusion_type

        self.linear1_temporal_h = nn.Linear(dim_hidden, dim_mid, bias=True)
        self.linear1_temporal_f = nn.Linear(dim_hidden, dim_mid, bias=True)
        self.linear2_temporal = nn.Linear(dim_mid if fusion_type == 'addition' else dim_mid * 2, 1, bias=False)

        self.with_multimodal_attention = with_multimodal_attention
        if self.with_multimodal_attention:
            self.linear1_modality_h = nn.Linear(dim_hidden, dim_mid, bias=True)
            self.linear1_modality_f = nn.Linear(dim_hidden, dim_mid, bias=True)
            self.linear2_modality = nn.Linear(dim_mid if fusion_type == 'addition' else dim_mid * 2, 1, bias=False)

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

    def modality_cal_out(self, linear1_list, linear2, input_list):
        assert isinstance(linear1_list, list) and isinstance(input_list, list)
        assert len(linear1_list) == len(input_list)

        res = []
        for i in range(len(input_list)):
            feat = input_list[i]
            assert len(feat.shape) == 2
            linear1_output = linear1_list[i](feat)
            res.append(self.activation(linear1_output) if self.activation_type == 'split' else linear1_output)
        
        if self.fusion_type == 'addition':
            output = torch.stack(res).sum(0)
        else:
            output = torch.cat(res, dim=1)

        if self.activation_type != 'split':
            output = self.activation(output)

        output = linear2(output)
        return output

    def forward(self, hidden_state, feats, enhance_feats=None, category=None, t=None):
        """
        feats: [batch_size, seq_len, dim]
        """

        if len(hidden_state.shape) == 1:
            hidden_state = hidden_state.unsqueeze(0)
        if len(hidden_state.shape) == 3 and hidden_state.shape[0] == 1:
            hidden_state = hidden_state.squeeze(0)

        context = []
        all_weight = []
        for i in range(self.num_feats):
            _, weight = self.cal_out(
                [self.linear1_temporal_h, self.linear1_temporal_f],
                self.linear2_temporal,
                [hidden_state, feats[i]]
                )
            all_weight.append(weight)
            context.append(torch.bmm(weight.unsqueeze(1), feats[i]).squeeze(1))

        if not self.with_multimodal_attention:
            final_context = torch.cat(context, dim=1)
            return final_context, torch.cat(all_weight, dim=1)

        weight_modality = []
        for i in range(self.num_feats):
            out = self.modality_cal_out(
                [self.linear1_modality_h, self.linear1_modality_f],
                self.linear2_modality,
                [hidden_state, context[i]]
                )
            weight_modality.append(out)
        weight_modality = F.softmax(torch.cat(weight_modality, dim=1), dim=1)
        final_context = torch.bmm(weight_modality.unsqueeze(1), torch.stack(context, dim=1)).squeeze(1)

        return final_context, weight_modality


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.head = head
        self.d_k = d_model // head
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, attention_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_k, self.head

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
        attention_scores = attention_scores / math.sqrt(self.d_k)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        #print("after att:", attention_probs)
        attention_probs = self.dropout(attention_probs)

        outputs = torch.bmm(attention_probs, v)

        outputs = outputs.view(n_head, sz_b, len_q, d_v)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        return outputs, attention_probs.view(n_head, sz_b, len_q, len_k)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.feed_foward = nn.Linear(size, size)
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        sublayer_outputs, others = sublayer(self.norm(x))
        y = self.dropout(self.feed_foward(self.dropout(sublayer_outputs)))
        return x + y, others

class Multi_Scale_Context_Attention(nn.Module):
    def __init__(self, opt):
        super(Multi_Scale_Context_Attention, self).__init__()

        self.mha = MultiHeadAttention(d_model=opt['dim_hidden'], head=1, dropout=0.1)
        self.modality = opt['modality'].lower()
        self.sublayer = SublayerConnection(size=opt['dim_hidden'], dropout=0.3)
        self.query_all = opt.get('query_all', False)

        self.with_gate = opt.get('with_gate', False)
        if self.with_gate:
            if len(self.modality) == 3:
                self.w_1 = nn.Linear(opt['dim_hidden'], opt['dim_hidden'])
                self.w_2 = nn.Linear(opt['dim_hidden'], opt['dim_hidden'])

    def forward(self, encoder_outputs):
        if self.with_gate:
            '''
            assert len(encoder_outputs) > 1
            pos = self.modality.index('i')
            image_feats = encoder_outputs[pos]

            attend_to_image_feats = []
            for i in range(len(encoder_outputs)):
                if i == pos:
                    continue
                outputs, others = self.sublayer(image_feats, lambda x: self.mha(encoder_outputs[i], x, x))
                attend_to_image_feats.append(outputs)

            if len(attend_to_image_feats) == 2:
                gate = torch.sigmoid(self.w_1(attend_to_image_feats[0]) + self.w_2(attend_to_image_feats[1]))
                outputs = gate * attend_to_image_feats[0] + (1 - gate) * attend_to_image_feats[0]
            else:
                outputs = attend_to_image_feats[0]
            return outputs, outputs.mean(1), others
            '''

            outputs = []
            hiddens = []
            others = []
            for j in range(len(encoder_outputs)):
                curr_feats = encoder_outputs[j]
                attend_to_curr_feats = []
                for i in range(len(encoder_outputs)):
                    if i == j:
                        continue
                    output, other = self.sublayer(curr_feats, lambda x: self.mha(encoder_outputs[i], x, x))
                    attend_to_curr_feats.append(output)
                    others.append(other)
                attend_to_curr_feats = torch.cat(attend_to_curr_feats, dim=1)
                outputs.append(attend_to_curr_feats)
                hiddens.append(attend_to_curr_feats.mean(1))
            return outputs, torch.stack(hiddens, dim=0).mean(0), torch.cat(others, dim=0)


        elif self.query_all:
            outputs = []
            hiddens = []
            others = []
            for j in range(len(encoder_outputs)):
                curr_feats = encoder_outputs[j]
                other_feats = []
                for i in range(len(encoder_outputs)):
                    if i == j:
                        continue
                    other_feats.append(encoder_outputs[i])
                other_feats = torch.cat(other_feats, dim=1)
                output, other = self.sublayer(curr_feats, lambda x: self.mha(x, other_feats, other_feats))
                
                outputs.append(output)
                hiddens.append(output.mean(1))
                others.append(other)
            return outputs, torch.stack(hiddens, dim=0).mean(0), torch.cat(others, dim=0)

        else:
            pos = self.modality.index('i')

            image_feats = encoder_outputs[pos]
            other_feats = []
            for i in range(len(encoder_outputs)):
                if i == pos:
                    continue
                other_feats.append(encoder_outputs[i])
            other_feats = torch.cat(other_feats, dim=1)

            outputs, others = self.sublayer(image_feats, lambda x: self.mha(x, other_feats, other_feats))

            return outputs, outputs.mean(1), others

'''
class Encoder_Baseline(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], encoder_type='gru', together=False):
        super(Encoder_Baseline, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = nn.Dropout(0.5)

        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        self.encoder = []

        for i in range(len(self.input_size)):
            if input_size[i] is None:
                self.encoder.append(None)
            else:
                tmp_module = GSRU(input_size[i], input_size[(i+1)%len(input_size)], output_size[i])
                self.add_module("Encoder%s"%(name[i]), tmp_module)
                self.encoder.append(tmp_module)

    def forward(self, input_feats):
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

        outputs = []
        hiddens = []

        for i in range(len(input_feats)):
            if self.encoder[i] is None:
                eo = input_feats[i]
                eh = eo.mean(1)
            else:
                eo, eh = self.encoder[i](self.dropout(input_feats[i]), self.dropout(input_feats[(i+1)%len(input_feats)]))

            outputs.append(eo)
            hiddens.append(eh)

        return outputs, hiddens

class GSRU(torch.nn.Module):
    def extra_repr(self):
        s = '{input_size}, {feats_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        return s.format(**self.__dict__)

    def __init__(self, input_size, feats_size, hidden_size, bias=True):
        super(GSRU, self).__init__()
        self.input_size = input_size
        self.feats_size = feats_size
        self.hidden_size = hidden_size
        self.bias = bias
        num_gates = 3
        self.weight_ih = Parameter(torch.Tensor(num_gates * hidden_size, input_size))
        self.weight_fh = Parameter(torch.Tensor((num_gates-1) * hidden_size, feats_size))
        self.weight_hh = Parameter(torch.Tensor(num_gates * hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.Tensor(num_gates * hidden_size))
        self.bias_fh = Parameter(torch.Tensor((num_gates-1) * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(num_gates * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward_each_timestep(self, input, feats, hx, weight_ih, weight_fh, weight_hh, bias_ih, bias_fh, bias_hh):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        if len(hx.shape) == 3:
            hx = hx.squeeze(0)
        #print(input.shape, feats.shape, hx.shape)
        assert input.size(1) == self.input_size
        assert feats.size(1) == self.feats_size
        assert hx.size(1) == self.hidden_size

        gi = F.linear(input, weight_ih, bias_ih)
        gf = F.linear(feats, weight_fh, bias_fh)
        gh = F.linear(hx, weight_hh, bias_hh)
        
        
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
        f_r, f_i = gf.chunk(2, 1)

        resetgate = F.sigmoid(i_r + h_r + f_r)
        inputgate = F.sigmoid(i_i + h_i + f_i)
        #selectgate = F.sigmoid(i_s + h_s + f_s)
        #newgate = F.tanh(i_n + (selectgate) * f_n + resetgate * h_n)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)
        
        return hy

    def forward(self, last_level_outputs, this_level_feats, hx=None):
        batch_size, seq_len, _ = last_level_outputs.shape
        output = []
        h = None if hx is None else hx.clone()

        for i in range(seq_len):
            input = last_level_outputs[:, i, :]
            feats = this_level_feats[:, i, :]
            hy = self.forward_each_timestep(input, feats, hx, 
                self.weight_ih, self.weight_fh, self.weight_hh, 
                self.bias_ih, self.bias_fh, self.bias_hh)
            output.append(hy.clone())
            hx = hy
        output = torch.stack(output, dim=1)
        return output, hx
'''

'''
class Encoder_Baseline(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], encoder_type='gru', together=False):
        super(Encoder_Baseline, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = nn.Dropout(0.5)

        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        self.encoder1 = []
        self.encoder2 = []

        for i in range(len(self.input_size)):
            if input_size[i] is None:
                self.encoder.append(None)
            else:
                tmp_module1 = nn.GRU(input_size[i], output_size[i], batch_first=True)
                tmp_module2 = GSRU(output_size[i], output_size[(i+1)%len(input_size)], output_size[i])
                self.add_module("Encoder%s_1"%(name[i]), tmp_module1)
                self.encoder1.append(tmp_module1)
                self.add_module("Encoder%s_2"%(name[i]), tmp_module2)
                self.encoder2.append(tmp_module2)

    def forward(self, input_feats):
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

        outputs = []
        hiddens = []

        for i in range(len(input_feats)):
            eo, eh = self.encoder1[i](self.dropout(input_feats[i]))
            outputs.append(eo)
            hiddens.append(eh)

        outputs2 = []
        hiddens2 = []
        for i in range(len(input_feats)):
            eo, eh = self.encoder2[i](self.dropout(outputs[i]), self.dropout(outputs[(i+1)%len(input_feats)]), hx=hiddens[i][0, :, :])
            outputs2.append(eo)
            hiddens2.append(eh)

        return outputs2, hiddens2

class GSRU(torch.nn.Module):
    def extra_repr(self):
        s = '{input_size}, {feats_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        return s.format(**self.__dict__)

    def __init__(self, input_size, feats_size, hidden_size, bias=True):
        super(GSRU, self).__init__()
        self.input_size = input_size
        self.feats_size = feats_size
        self.hidden_size = hidden_size
        self.bias = bias
        num_gates = 3
        self.weight_ih = Parameter(torch.Tensor(num_gates * hidden_size, input_size))
        self.weight_fh = None
        self.weight_hh = Parameter(torch.Tensor(num_gates * hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.Tensor(num_gates * hidden_size))
        self.bias_fh = None
        self.bias_hh = Parameter(torch.Tensor(num_gates * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward_each_timestep(self, input, feats, hx, weight_ih, weight_fh, weight_hh, bias_ih, bias_fh, bias_hh):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        if len(hx.shape) == 3:
            hx = hx.squeeze(0)
        #print(input.shape, feats.shape, hx.shape)
        assert input.size(1) == self.input_size
        assert feats.size(1) == self.feats_size
        assert hx.size(1) == self.hidden_size

        gi = F.linear(input, weight_ih, bias_ih)
        gh = F.linear(hx, weight_hh, bias_hh)
        
        
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r + feats)
        inputgate = F.sigmoid(i_i + h_i + feats)
        #selectgate = F.sigmoid(i_s + h_s + f_s)
        #newgate = F.tanh(i_n + (selectgate) * f_n + resetgate * h_n)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)
        
        return hy

    def forward(self, last_level_outputs, this_level_feats, hx=None):
        batch_size, seq_len, _ = last_level_outputs.shape
        output = []
        h = None if hx is None else hx.clone()

        for i in range(seq_len):
            input = last_level_outputs[:, i, :]
            feats = this_level_feats[:, i, :]
            hy = self.forward_each_timestep(input, feats, hx, 
                self.weight_ih, self.weight_fh, self.weight_hh, 
                self.bias_ih, self.bias_fh, self.bias_hh)
            output.append(hy.clone())
            hx = hy
        output = torch.stack(output, dim=1)
        return output, hx
'''


class Encoder_Baseline2(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], encoder_type='gru', together=False):
        super(Encoder_Baseline2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        print(input_size)
        print(output_size)
        self.encoder1 = Hierarchical_Encoder(input_size, output_size, {'encoder_dropout': 0.5})
        input_size.reverse()
        self.encoder2 = Hierarchical_Encoder(input_size, output_size, {'encoder_dropout': 0.5})

    def forward(self, input_feats):
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

        outputs = []
        hiddens = []

        
        eo, eh = self.encoder1(input_feats)
        outputs.append(eo)
        hiddens.append(eh)

        input_feats.reverse()
        eo2, eh2 = self.encoder2(input_feats)
        outputs.append(eo2)
        hiddens.append(eh2)

        return outputs, hiddens

'''
class Encoder_Baseline(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], encoder_type='gru', together=False, opt=None):
        super(Encoder_Baseline, self).__init__()
        self.input_size = input_size
        self.hidden_size = output_size
        self.input_dropout_p = opt['encoder_dropout']
        self.dropout = nn.Dropout(self.input_dropout_p)

        rnn = nn.GRU

        for i in range(len(input_size)):
            tmp = self.hidden_size[i-1] + self.input_size[i] if i else self.input_size[i]
            tmpRNN = rnn(tmp, self.hidden_size[i], batch_first=True, num_layers=1)
            self.add_module("rnn%d"%i, tmpRNN)
        self.rnn_list = []
        for name, module in self.named_children():
            if 'rnn' in name: self.rnn_list.append(module)



    def forward(self, input_feats):
        assert len(input_feats) == len(self.input_size)
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len


        encoder_hidden = None
        rnn_outputs = []
        encoder_hiddens = []ats[i]], dim=2)
                rnn_output, encoder_hidden = self.rnn_list[i](self.dropout(input), encoder_hidden)
            else:
                rnn_output, encoder_hidden = self.rnn_list[i](self.dropout(input_feats[i]), encoder_hidden)
            
            rnn_outputs.append(rnn_output.clone
        for i in range(len(input_feats)):
            if i:
                input = torch.cat([rnn_output, input_fe())
            encoder_hiddens.append(encoder_hidden[-1, :, :].clone())

        return rnn_outputs, encoder_hiddens
'''

class base_unit(nn.Module):
    def __init__(self, input_size, output_size):
        super(base_unit, self).__init__()
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)  

    def forward(self, input_feats):
        feats = self.linear(input_feats)
        bsz, seq_len, _ = feats.shape
        feats = self.bn(feats.contiguous().view(bsz * seq_len, -1)).view(bsz, seq_len, self.output_size)
        return feats

'''
class Encoder_Baseline(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], encoder_type='gru', together=False):
        super(Encoder_Baseline, self).__init__()
        self.input_size = input_size
        self.num = len(input_size)
        self.output_size = output_size
        self.dropout = nn.Dropout(0.5)

        self.emb = []


        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        self.encoder = []

        for i in range(self.num):
            tmp_emb = base_unit(input_size[i], output_size[i])
            tmp_module = GSRU(input_size[i], [512] * (self.num - 1), output_size[i])
            self.add_module("Emb%s"%name[i], tmp_emb)
            self.add_module("Encoder%s"%(name[i]), tmp_module)
            self.emb.append(tmp_emb)
            self.encoder.append(tmp_module)

        

    def forward(self, input_feats):
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

        input_feats = [self.dropout(item) for item in input_feats]

        embedding = []
        for i in range(len(input_feats)):
            embedding.append(self.emb[i](input_feats[i]))

        outputs = []
        hiddens = []
        eh = None
        for i in range(len(input_feats)):
            this_input_feats = input_feats[i]
            other_input_embs = []
            for j in range(self.num):
                if j != i:
                    other_input_embs.append(embedding[j])
            
            eo, eh = self.encoder[i](this_input_feats, other_input_embs, hx=eh)

            outputs.append(eo)
            hiddens.append(eh)

        return outputs, hiddens

class GSRU(torch.nn.Module):
    def extra_repr(self):
        s = '{input_size}, {feats_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        return s.format(**self.__dict__)

    def __init__(self, input_size, feats_size, hidden_size, bias=True):
        super(GSRU, self).__init__()
        self.input_size = input_size
        self.feats_size = feats_size
        self.hidden_size = hidden_size
        self.bias = bias
        num_gates = 3
        self.weight_ih = Parameter(torch.Tensor(num_gates * hidden_size, input_size))
        self.weight_fh = Parameter(torch.Tensor((num_gates-1) * hidden_size, hidden_size))
        self.weight_hh = Parameter(torch.Tensor(num_gates * hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.Tensor(num_gates * hidden_size))
        self.bias_fh = Parameter(torch.Tensor((num_gates-1) * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(num_gates * hidden_size))

        #if len(feats_size) > 1:
        #    self.select_h = nn.Linear(hidden_size, hidden_size)
        #    self.select_f = nn.Linear(hidden_size, hidden_size)
        #    self.score = nn.Linear(hidden_size, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward_each_timestep(self, input, feats, hx, weight_ih, weight_fh, weight_hh, bias_ih, bias_fh, bias_hh):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        if len(hx.shape) == 3:
            hx = hx.squeeze(0)
        #print(input.shape, feats.shape, hx.shape)
        assert input.size(1) == self.input_size
        assert hx.size(1) == self.hidden_size

        gi = F.linear(input, weight_ih, bias_ih)
        #assert len(feats) == 1

        context = torch.stack(feats, 0).sum(0)
        gf = F.linear(context, weight_fh, bias_fh)
        gh = F.linear(hx, weight_hh, bias_hh)
        
        
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
        f_r, f_i = gf.chunk(2, 1)

        resetgate = F.sigmoid(i_r + h_r + f_r)
        inputgate = F.sigmoid(i_i + h_i + f_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)
        
        return hy

    def forward(self, last_level_outputs, this_level_feats, hx=None):
        batch_size, seq_len, _ = last_level_outputs.shape
        output = []
        h = None if hx is None else hx.clone()

        for i in range(seq_len):
            input = last_level_outputs[:, i, :]
            feats = [item[:, i, :] for item in this_level_feats]
            hy = self.forward_each_timestep(input, feats, hx, 
                self.weight_ih, self.weight_fh, self.weight_hh, 
                self.bias_ih, self.bias_fh, self.bias_hh)
            output.append(hy.clone())
            hx = hy
        output = torch.stack(output, dim=1)
        return output, hx
'''

'''
class Encoder_Baseline(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], encoder_type='gru', together=False):
        super(Encoder_Baseline, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = nn.Dropout(0.5)

        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        self.encoder = []

        for i in range(len(self.input_size)):
            if input_size[i] is None:
                self.encoder.append(None)
            else:
                tmp_module = myGRU(input_size[i], output_size[i])
                self.add_module("Encoder%s"%(name[i]), tmp_module)
                self.encoder.append(tmp_module)

    def forward(self, input_feats):
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

        outputs = []
        hiddens = []

        for i in range(len(input_feats)):
            eo, eh = self.encoder[i](self.dropout(input_feats[i]))
            outputs.append(eo)
            hiddens.append(eh)

        return outputs, hiddens
'''
class myGRU(torch.nn.Module):
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        return s.format(**self.__dict__)

    def __init__(self, input_size, hidden_size, bias=True):
        super(myGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        num_gates = 3
        self.weight_ih = Parameter(torch.Tensor(num_gates * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(num_gates * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(num_gates * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(num_gates * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward_each_timestep(self, input, hx, weight_ih, weight_hh, bias_ih, bias_hh):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        if len(hx.shape) == 3:
            hx = hx.squeeze(0)
        #print(input.shape, feats.shape, hx.shape)
        assert input.size(1) == self.input_size
        assert hx.size(1) == self.hidden_size

        gi = F.linear(input, weight_ih, bias_ih)
        gh = F.linear(hx, weight_hh, bias_hh)
        
        
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)

        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)
        
        return hy

    def forward(self, feats, hx=None):
        batch_size, seq_len, _ = feats.shape
        output = []
        h = None if hx is None else hx.clone()

        for i in range(seq_len):
            input = feats[:, i, :]
            hy = self.forward_each_timestep(input, hx, 
                self.weight_ih, self.weight_hh, 
                self.bias_ih, self.bias_hh)
            output.append(hy.clone())
            hx = hy
        output = torch.stack(output, dim=1)
        return output, hx


'''
class Encoder_Baseline(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], encoder_type='gru', together=False):
        super(Encoder_Baseline, self).__init__()
        self.input_size = input_size
        self.num = len(input_size)
        self.output_size = output_size
        self.dropout = nn.Dropout(0.5)

        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        self.encoder = []

        for i in range(self.num):
            tmp_module = myGRU(input_size[i], output_size[i])
            self.add_module("Encoder%s"%(name[i]), tmp_module)
            self.encoder.append(tmp_module)

        

    def forward(self, input_feats):
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

        input_feats = [self.dropout(item) for item in input_feats]

        outputs = []
        hiddens = []
        eh = None
        for i in range(len(input_feats)):
            this_input_feats = input_feats[i]
            eo, eh = self.encoder[i](this_input_feats, hx=eh)
            outputs.append(eo)
            hiddens.append(eh)

        return outputs, hiddens
'''

class Encoder_Baseline(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], auxiliary_pos=[[], [0]], skip_info=[1, 0], return_gate_info=False, use_LSTM=False):
        super(Encoder_Baseline, self).__init__()
        assert len(input_size) == len(output_size)
        assert len(input_size) == len(auxiliary_pos)
        assert len(input_size) == len(skip_info)
        assert sum(skip_info) != len(skip_info)

        self.input_size = input_size
        self.output_size = output_size
        self.auxiliary_pos = auxiliary_pos
        self.dropout = nn.Dropout(0.5)
        self.return_gate_info = return_gate_info

        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        self.encoder = []
        for i in range(len(input_size)):
            if skip_info[i]:
                self.encoder.append(None)
            else:
                auxiliary_size = []
                for pos in self.auxiliary_pos[i]:
                    auxiliary_size.append(input_size[pos])
                rnn = GRU_with_GCC if not use_LSTM else LSTM_with_GCC
                tmp_module = rnn(input_size[i], auxiliary_size, output_size[i], return_gate_info=return_gate_info)
                self.encoder.append(tmp_module)
                self.add_module('Encoder%s'%name[i], tmp_module)

    def will_return_gate_info(self):
        return self.return_gate_info

    def forward(self, input_feats):
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

        input_feats = [self.dropout(item) for item in input_feats]

        outputs = []
        hiddens = []
        gate = []
        for i in range(len(input_feats)):
            if self.encoder[i] is None:
                continue

            auxiliary_feats = []
            for pos in self.auxiliary_pos[i]:
                auxiliary_feats.append(input_feats[pos])

            if self.return_gate_info:
                eo, eh, g = self.encoder[i](input_feats[i], auxiliary_feats, hx=None)
                gate.append(g)
            else:
                eo, eh = self.encoder[i](input_feats[i], auxiliary_feats, hx=None)
            outputs.append(eo)
            hiddens.append(eh)

        if self.return_gate_info:
            return outputs, hiddens, gate
        else:
            return outputs, hiddens

class GRU_with_GCC(torch.nn.Module):
    def extra_repr(self):
        s = '{input_size}, {feats_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        return s.format(**self.__dict__)

    def __init__(self, input_size, feats_size, hidden_size, bias=True, return_gate_info=False):
        super(GRU_with_GCC, self).__init__()
        assert isinstance(feats_size, list)
        self.input_size = input_size
        self.feats_size = feats_size
        self.hidden_size = hidden_size
        self.bias = bias
        num_gates = 3
        self.num_gates = 3
        self.weight_ih = Parameter(torch.Tensor(num_gates * hidden_size, input_size))
        self.weight_fh = nn.ParameterList([Parameter(torch.Tensor((num_gates-1) * hidden_size, fsize)) for fsize in feats_size])
        self.weight_hh = Parameter(torch.Tensor(num_gates * hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.Tensor(num_gates * hidden_size))
        self.bias_fh = nn.ParameterList([Parameter(torch.Tensor((num_gates-1) * hidden_size)) for _ in range(len(feats_size))])
        self.bias_hh = Parameter(torch.Tensor(num_gates * hidden_size))

        self.return_gate_info = return_gate_info

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward_each_timestep(self, input, feats, hx, weight_ih, weight_fh, weight_hh, bias_ih, bias_fh, bias_hh):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        if len(hx.shape) == 3:
            hx = hx.squeeze(0)

        assert input.size(1) == self.input_size
        assert hx.size(1) == self.hidden_size

        gi = F.linear(input, weight_ih, bias_ih)
        gh = F.linear(hx, weight_hh, bias_hh)

        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        gf = []
        for i in range(len(feats)):
            gf.append(F.linear(feats[i], weight_fh[i], bias_fh[i]))
        if len(gf): 
            gf = torch.stack(gf, dim=0).sum(0)
            #gf = torch.stack(gf, dim=0).mean(0)
            f_r, f_i = gf.chunk(2, 1)
            resetgate = F.sigmoid(i_r + h_r + f_r)
            inputgate = F.sigmoid(i_i + h_i + f_i)
        else:
            resetgate = F.sigmoid(i_r + h_r)
            inputgate = F.sigmoid(i_i + h_i)

        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)
        
        return hy, [resetgate, inputgate]

    def forward(self, input_feats, auxiliary_feats, hx=None):
        batch_size, seq_len, _ = input_feats.shape
        output = []
        gate = [[] for _ in range(self.num_gates-1)]
        h = None if hx is None else hx.clone()

        for i in range(seq_len):
            input = input_feats[:, i, :]
            feats = [item[:, i, :] for item in auxiliary_feats]
            hy, g = self.forward_each_timestep(input, feats, hx, 
                self.weight_ih, self.weight_fh, self.weight_hh, 
                self.bias_ih, self.bias_fh, self.bias_hh)
            output.append(hy.clone())
            if self.return_gate_info:
                for j, item in enumerate(g):
                    gate[j].append(item)
            hx = hy

        output = torch.stack(output, dim=1)
        if self.return_gate_info:
            for j in range(self.num_gates-1):
                gate[j] = torch.stack(gate[j], dim=1)
            return output, hx, gate
        return output, hx


class LSTM_with_GCC(torch.nn.Module):
    def extra_repr(self):
        s = '{input_size}, {feats_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        return s.format(**self.__dict__)

    def __init__(self, input_size, feats_size, hidden_size, bias=True, return_gate_info=False):
        super(LSTM_with_GCC, self).__init__()
        assert isinstance(feats_size, list)
        self.input_size = input_size
        self.feats_size = feats_size
        self.hidden_size = hidden_size
        self.bias = bias
        num_gates = 4
        self.num_gates = 4
        self.weight_ih = Parameter(torch.Tensor(num_gates * hidden_size, input_size))
        self.weight_fh = nn.ParameterList([Parameter(torch.Tensor((num_gates-1) * hidden_size, fsize)) for fsize in feats_size])
        self.weight_hh = Parameter(torch.Tensor(num_gates * hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.Tensor(num_gates * hidden_size))
        self.bias_fh = nn.ParameterList([Parameter(torch.Tensor((num_gates-1) * hidden_size)) for _ in range(len(feats_size))])
        self.bias_hh = Parameter(torch.Tensor(num_gates * hidden_size))

        self.return_gate_info = return_gate_info

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward_each_timestep(self, input, feats, hidden, weight_ih, weight_fh, weight_hh, bias_ih, bias_fh, bias_hh):
        assert input.size(1) == self.input_size
        if hidden is None:
            hidden = (input.new_zeros(input.size(0), self.hidden_size, requires_grad=False), input.new_zeros(input.size(0), self.hidden_size, requires_grad=False))

        hx, cx = hidden
        gates = F.linear(input, weight_ih, bias_ih) + F.linear(hx, weight_hh, bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)


        tmp = []
        for i in range(len(feats)):
            tmp.append(F.linear(feats[i], weight_fh[i], bias_fh[i]))

        if len(tmp):
            tmp = torch.stack(tmp, dim=0).sum(0)
            tmpi, tmpf, tmpo = tmp.chunk(3, 1)
            ingate += tmpi
            forgetgate += tmpf
            outgate += tmpo

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy, [ingate, forgetgate, outgate]

    def forward(self, input_feats, auxiliary_feats, hx=None):
        batch_size, seq_len, _ = input_feats.shape
        output = []
        gate = [[] for _ in range(self.num_gates-1)]
        
        
        for i in range(seq_len):
            input = input_feats[:, i, :]
            feats = [item[:, i, :] for item in auxiliary_feats]
            hy, cy, g = self.forward_each_timestep(input, feats, hx, 
                self.weight_ih, self.weight_fh, self.weight_hh, 
                self.bias_ih, self.bias_fh, self.bias_hh)

            hx = (hy, cy)
            output.append(hy.clone())

            if self.return_gate_info:
                for j, item in enumerate(g):
                    gate[j].append(item)


        output = torch.stack(output, dim=1)
        if self.return_gate_info:
            for j in range(self.num_gates-1):
                gate[j] = torch.stack(gate[j], dim=1)
            return output, hx, gate
        return output, hx


'''
class Encoder_Baseline(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], auxiliary_pos=[[], [0]], skip_info=[1, 0], return_gate_info=False, use_LSTM=False):
        super(Encoder_Baseline, self).__init__()
        assert len(input_size) == len(output_size)
        assert len(input_size) == len(auxiliary_pos)
        assert len(input_size) == len(skip_info)
        assert sum(skip_info) != len(skip_info)

        self.input_size = input_size
        self.output_size = output_size
        self.auxiliary_pos = auxiliary_pos
        self.dropout = nn.Dropout(0.5)
        self.return_gate_info = return_gate_info

        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        self.weights = []
        self.bias = []
        self.linear_list = []
        for i in range(len(input_size)):
            ng = 3 if not skip_info[i] else 2
            self.register_parameter('shared_weight%d'%i, Parameter(torch.Tensor(ng * 512, input_size[i])))
            self.register_parameter('shared_bias%d'%i, Parameter(torch.Tensor(ng * 512)))
            
            #tmp_module = nn.Linear(input_size[i], ng*512)
            #self.add_module('linear%d'%i, tmp_module)
            #self.weights.append(tmp_module.weight)
            #self.bias.append(tmp_module.bias)
            

        stdv = 1.0 / math.sqrt(512)
        for k, v in self.named_parameters():
            v.data.uniform_(-stdv, stdv)
            if 'shared_weight' in k:
                self.weights.append(v)
            if 'shared_bias' in k:
                self.bias.append(v)

        self.encoder = []
        for i in range(len(input_size)):
            if skip_info[i]:
                self.encoder.append(None)
            else:
                rnn = GRU_with_GCC_with_shared_params
                tmp_module = rnn(output_size[i], return_gate_info=return_gate_info)
                self.encoder.append(tmp_module)
                self.add_module('Encoder%s'%name[i], tmp_module)


    def will_return_gate_info(self):
        return self.return_gate_info

    def forward(self, input_feats):
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

        input_feats = [self.dropout(item) for item in input_feats]

        outputs = []
        hiddens = []
        gate = []
        for i in range(len(input_feats)):
            if self.encoder[i] is None:
                continue

            auxiliary_feats = []
            wfh = []
            bfh = []
            for pos in self.auxiliary_pos[i]:
                auxiliary_feats.append(input_feats[pos])
                wfh.append(self.weights[pos])
                bfh.append(self.bias[pos])

            if self.return_gate_info:
                eo, eh, g = self.encoder[i](input_feats[i], auxiliary_feats, hx=None, 
                    weight_ih=self.weights[i], weight_fh=wfh, bias_ih=self.bias[i], bias_fh=bfh)
                gate.append(g)
            else:
                eo, eh = self.encoder[i](input_feats[i], auxiliary_feats, hx=None, 
                    weight_ih=self.weights[i], weight_fh=wfh, bias_ih=self.bias[i], bias_fh=bfh)
            outputs.append(eo)
            hiddens.append(eh)

        if self.return_gate_info:
            return outputs, hiddens, gate
        else:
            return outputs, hiddens


class Encoder_Baseline(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], auxiliary_pos=[[], [0]], skip_info=[1, 0], return_gate_info=False, use_LSTM=False, num_factor=512):
        super(Encoder_Baseline, self).__init__()
        assert len(input_size) == len(output_size)
        assert len(input_size) == len(auxiliary_pos)
        assert len(input_size) == len(skip_info)
        assert sum(skip_info) != len(skip_info)

        self.input_size = input_size
        self.output_size = output_size
        self.auxiliary_pos = auxiliary_pos
        self.dropout = nn.Dropout(0.5)
        self.return_gate_info = return_gate_info

        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)
        
        self.length_tag = []
        for i in range(len(input_size)):
            c = 1 if not skip_info[i] else 0
            for j in range(len(auxiliary_pos)):
                for pos in auxiliary_pos[j]:
                    if pos == i:
                        c += 1
            self.length_tag.append(c)

        self.shared_module = []
        for i in range(len(input_size)):
            num_gate = 3 if not skip_info[i] else 2
            
            tmp_module = SVD_Weights(input_size[i], output_size[i], num_gate=2, length_tag=self.length_tag[i], nf=num_factor, individual=(not skip_info[i]))
            self.add_module('Param%d'%i, tmp_module)
            self.shared_module.append(tmp_module)


        self.encoder = []
        for i in range(len(input_size)):
            if skip_info[i]:
                self.encoder.append(None)
            else:
                rnn = GRU_with_GCC_with_shared_params
                tmp_module = rnn(output_size[i], return_gate_info=return_gate_info)
                self.encoder.append(tmp_module)
                self.add_module('Encoder%s'%name[i], tmp_module)


    def will_return_gate_info(self):
        return self.return_gate_info

    def reset_status(self, device):
        length = len(self.length_tag)
        self.index = [0] * length
        self.all_tag = [[] for _ in range(length)]

        for i, item in enumerate(self.length_tag):
            for j in range(item):
                tag = torch.zeros(item, 1).to(device)
                tag[j] = 1
                self.all_tag[i].append(tag)

    def get_status(self, i):
        tag = self.all_tag[i][self.index[i]]
        self.index[i] += 1
        return tag

    def forward(self, input_feats):
        self.reset_status(input_feats[0].device)

        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

        input_feats = [self.dropout(item) for item in input_feats]

        outputs = []
        hiddens = []
        gate = []
        for i in range(len(input_feats)):
            if self.encoder[i] is None:
                continue

            major_w, major_b = self.shared_module[i].get_all_params(tag=self.get_status(i))

            auxiliary_feats = []
            wfh = []
            bfh = []
            for pos in self.auxiliary_pos[i]:
                auxiliary_feats.append(input_feats[pos])
                w, b = self.shared_module[pos].get_shared_params(tag=self.get_status(pos))
                wfh.append(w)
                bfh.append(b)

            if self.return_gate_info:
                eo, eh, g = self.encoder[i](input_feats[i], auxiliary_feats, hx=None, 
                    weight_ih=major_w, weight_fh=wfh, bias_ih=major_b, bias_fh=bfh)
                gate.append(g)
            else:
                eo, eh = self.encoder[i](input_feats[i], auxiliary_feats, hx=None, 
                    weight_ih=major_w, weight_fh=wfh, bias_ih=major_b, bias_fh=bfh)
            outputs.append(eo)
            hiddens.append(eh)

        if self.return_gate_info:
            return outputs, hiddens, gate
        else:
            return outputs, hiddens
'''


class GRU_with_GCC_with_shared_params(torch.nn.Module):
    def extra_repr(self):
        s = 'hidden={hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        return s.format(**self.__dict__)

    def __init__(self, hidden_size, bias=True, return_gate_info=False):
        super(GRU_with_GCC_with_shared_params, self).__init__()
        self.hidden_size = hidden_size
        self.bias = bias

        self.num_gates = 3
        self.weight_hh = Parameter(torch.Tensor(self.num_gates * hidden_size, hidden_size))
        self.bias_hh = Parameter(torch.Tensor(self.num_gates * hidden_size))
        self.return_gate_info = return_gate_info
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward_each_timestep(self, input, feats, hx, weight_ih, weight_fh, weight_hh, bias_ih, bias_fh, bias_hh):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        if len(hx.shape) == 3:
            hx = hx.squeeze(0)

        assert hx.size(1) == self.hidden_size

        gi = F.linear(input, weight_ih, bias_ih)
        gh = F.linear(hx, weight_hh, bias_hh)

        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        fr, fi = [], []
        for i in range(len(feats)):
            tmp = F.linear(feats[i], weight_fh[i], bias_fh[i])
            if tmp.shape[1] // self.hidden_size == self.num_gates:
                tmpr, tmpi, _ = tmp.chunk(3, 1)
            else:
                tmpr, tmpi = tmp.chunk(2, 1)
            fr.append(tmpr)
            fi.append(tmpi)

        if len(fr): 
            f_r = torch.stack(fr, dim=0).mean(0)
            f_i = torch.stack(fi, dim=0).mean(0)
            resetgate = F.sigmoid(i_r + h_r + f_r)
            inputgate = F.sigmoid(i_i + h_i + f_i)
        else:
            resetgate = F.sigmoid(i_r + h_r)
            inputgate = F.sigmoid(i_i + h_i)

        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)
        
        return hy, [resetgate, inputgate]

    def forward(self, input_feats, auxiliary_feats, hx, weight_ih, weight_fh, bias_ih, bias_fh):
        batch_size, seq_len, _ = input_feats.shape
        output = []
        gate = [[] for _ in range(self.num_gates-1)]
        h = None if hx is None else hx.clone()

        for i in range(seq_len):
            input = input_feats[:, i, :]
            feats = [item[:, i, :] for item in auxiliary_feats]
            hy, g = self.forward_each_timestep(input, feats, hx, 
                weight_ih, weight_fh, self.weight_hh, 
                bias_ih, bias_fh, self.bias_hh)
            output.append(hy.clone())
            if self.return_gate_info:
                for j, item in enumerate(g):
                    gate[j].append(item)
            hx = hy

        output = torch.stack(output, dim=1)
        if self.return_gate_info:
            for j in range(self.num_gates-1):
                gate[j] = torch.stack(gate[j], dim=1)
            return output, hx, gate
        return output, hx

class SVD_Weights(nn.Module):
    def __init__(self, input_size, output_size, num_gate, length_tag, nf=256, individual=False):
        super(SVD_Weights, self).__init__()
        self.length_tag = length_tag
        self.output_size = output_size

        self.weight_a = Parameter(torch.Tensor(num_gate * output_size, nf))
        self.weight_b = Parameter(torch.Tensor(nf, length_tag))
        self.weight_c = Parameter(torch.Tensor(nf, input_size))

        self.bias = Parameter(torch.Tensor(num_gate * output_size))

        if individual:
            self.individual_weight = Parameter(torch.Tensor(output_size, input_size))
            self.individual_bias = Parameter(torch.Tensor(output_size))
        else:
            self.individual_weight = None
            self.individual_bias = None

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.output_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)



    def get_shared_params(self, tag):
        assert tag.shape == (self.length_tag, 1)
        
        weight_mid = torch.mm(self.weight_b, tag)
        weight_mid = torch.diag(weight_mid.squeeze(1))
        weight = torch.mm(torch.mm(self.weight_a, weight_mid), self.weight_c)

        return weight, self.bias

    def get_individual_params(self):
        assert self.individual_weight is not None
        assert self.individual_bias is not None
        return self.individual_weight, self.individual_bias

    def get_all_params(self, tag):
        w1, b1 = self.get_shared_params(tag)
        w2, b2 = self.get_individual_params()

        return torch.cat([w1, w2], dim=0), torch.cat([b1, b2], dim=0)

'''
CUDA_VISIBLE_DEVICES=2 python train.py -ss -wc -all -m ami --scope base_Ami --skip_info 0 1 1 -afa mi
'''