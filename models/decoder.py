from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from fastNLP.modules import VarLSTM

class Embedding_Layer(nn.Module):
    def __init__(self, vocab_size, dim_word=300, dim_hidden=512, embedding_weights=None, train_embedding=False):
        super(Embedding_Layer, self).__init__()
        self.dim_hidden = dim_hidden

        self.emb = nn.Embedding(vocab_size, dim_word)
        if embedding_weights is not None:
            self.emb.from_pretrained(torch.FloatTensor(embedding_weights))
        if not train_embedding:
            print('===== Will not train word embeddings =====')
            for p in self.emb.parameters():
                p.requires_grad = False

        self.e2h = nn.Linear(dim_word, dim_hidden, bias=False)

    def forward(self, x):
        # x: [batch_size, seq_len] LongTensor
        x = self.emb(x) # [batch_size, seq_len, dim_word]
        x = self.e2h(x) # [batch_size, seq_len, dim_hidden]
        return x

    def linear(self, x):
        # x: [batch_size, dim_hidden]
        x = x.matmul(self.e2h.weight) # [batch_size, dim_word]
        x = x.matmul(self.emb.weight.t()) # [batch_size, vocab_size]
        return x

class Embedding_Layer_ne2h(nn.Module):
    def __init__(self, vocab_size, dim_word=512, embedding_weights=None, train_embedding=True, use_LN=False):
        super(Embedding_Layer_ne2h, self).__init__()

        self.emb = nn.Embedding(vocab_size, dim_word)
        if embedding_weights is not None:
            self.emb.from_pretrained(torch.FloatTensor(embedding_weights), train_embedding)
        self.ln = nn.LayerNorm(dim_word) if use_LN else None


    def forward(self, x):
        # x: [batch_size, seq_len] LongTensor
        #print(x.max(), x.min())
        x = self.emb(x) # [batch_size, seq_len, dim_word]
        return x if self.ln is None else self.ln(x)

    def linear(self, x):
        # x: [batch_size, dim_hidden]
        x = x.matmul(self.emb.weight.t()) # [batch_size, vocab_size]
        return x



class LSTM_Decoder(nn.Module):
    def __init__(self, opt, embedding=None, num_modality=-1):
        super(LSTM_Decoder, self).__init__()
        with_multimodal_attention = opt.get('with_multimodal_attention', False)
        addition = opt.get('addition', False)
        temporal_concat = opt.get('temporal_concat', False)
        gated_sum = opt.get('gated_sum', False)
        bidirectional = opt.get('bidirectional', False)

        if num_modality == -1:
            num_modality = 1 if (opt['encoder_type'] == 'IPE' or addition or gated_sum or temporal_concat) else (len(opt['modality']) - sum(opt['skip_info']))


        self.word_size = opt['dim_hidden']
        if opt['encoder_type'] == 'IPE' or with_multimodal_attention or addition or gated_sum or temporal_concat:
            self.feats_size = opt['dim_hidden']  
        else:
            self.feats_size = opt['dim_hidden'] * num_modality

        if bidirectional:
            self.feats_size *= 2

        self.hidden_size = opt["dim_hidden"]
        self.vocab_size = opt["vocab_size"]
        self.max_len = opt['max_len']

        if opt.get('others', False):
            import pickle
            self.embedding = Embedding_Layer(
                    vocab_size = self.vocab_size, 
                    dim_word = 300, 
                    dim_hidden = self.hidden_size, 
                    embedding_weights = pickle.load(open(opt['corpus_pickle'], 'rb'))['glove'], 
                    train_embedding=False
                )
        else:
            self.embedding = embedding if embedding is not None else nn.Embedding(self.vocab_size, self.word_size)


        lstm_input_size = self.word_size + self.feats_size + (opt['num_category'] if opt['with_category'] else 0)
        

        self.with_category = opt['with_category']
        self.rnn = nn.LSTMCell(lstm_input_size, self.hidden_size)
        self.forget_bias = opt.get('forget_bias', 0.6)
        self._init_lstm_forget_bias()

        self.att = Attentional_Attention(
                self.hidden_size, 
                [self.hidden_size * (2 if bidirectional else 1)] * num_modality, 
                opt.get('att_mid_size', 256),
                with_multimodal_attention=with_multimodal_attention,
                bidirectional=bidirectional
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


    def forward(self, **kwargs):
        # get the infomation we need from the kwargs
        it, encoder_outputs, category, decoder_hidden = map(
            lambda x: kwargs[x], 
            ["it", "encoder_outputs", "category", "decoder_hidden"]
        )

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

        return {
            'dec_outputs': self.dropout(decoder_hidden[0]),
            'dec_hidden': decoder_hidden,
            'weights': frames_weight
        }

class LSTM_GCC_Decoder(nn.Module):
    def __init__(self, opt, embedding=None, num_modality=-1):
        super(LSTM_GCC_Decoder, self).__init__()
        with_multimodal_attention = opt.get('with_multimodal_attention', False)
        addition = opt.get('addition', False)

        if num_modality == -1:
            num_modality = 1 if (opt['encoder_type'] == 'IPE' or addition) else (len(opt['modality']) - sum(opt['skip_info']))

        self.word_size = opt['dim_hidden']
        if opt['encoder_type'] == 'IPE' or with_multimodal_attention or addition:
            self.feats_size = opt['dim_hidden']  
        else:
            self.feats_size = opt['dim_hidden'] * num_modality

        self.hidden_size = opt["dim_hidden"]
        self.vocab_size = opt["vocab_size"]
        self.max_len = opt['max_len']

        self.embedding = embedding if embedding is not None else nn.Embedding(self.vocab_size, self.word_size)

        input_size = self.word_size
        auxiliary_size = self.feats_size + (opt['num_category'] if opt['with_category'] else 0)
        

        self.with_category = opt['with_category']
        self.rnn = LSTM_with_GCC_Cell(input_size, auxiliary_size, self.hidden_size)
        self.forget_bias = opt.get('forget_bias', 0.6)
        #self._init_lstm_forget_bias()

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
            auxiliary = torch.cat([context, category], dim=1)
        else:
            auxiliary = context

        decoder_hidden, gate = self.rnn(self.dropout(current_words), self.dropout(auxiliary), decoder_hidden)

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



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Attentional_Attention(nn.Module):
    def __init__(self, 
        dim_hidden, dim_feats, dim_mid, 
        activation=F.tanh, activation_type='acc', fusion_type='addition', with_multimodal_attention=False, different_wf=False, bidirectional=False):
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

        self.different_wf = different_wf
        if different_wf:
            self.linear1_temporal_f = []
            for i, item in enumerate(dim_feats):
                tmp_module = nn.Linear(item, dim_mid)
                self.add_module('linear1_temporal_f%d'%i, tmp_module)
                self.linear1_temporal_f.append(tmp_module)
        else:
            self.linear1_temporal_f = nn.Linear(dim_feats[0] * 2 if bidirectional else dim_feats[0], dim_mid, bias=True)
        #self.linear1_temporal_f = clones(nn.Linear(dim_hidden, dim_mid, bias=True), self.num_feats)
        self.linear2_temporal = nn.Linear(dim_mid if fusion_type == 'addition' else dim_mid * 2, 1, bias=False)

        self.with_multimodal_attention = with_multimodal_attention
        if self.with_multimodal_attention:
            self.linear1_modality_h = nn.Linear(dim_hidden, dim_mid, bias=True)
            self.linear1_modality_f = nn.Linear(dim_feats[0], dim_mid, bias=True)
            #self.linear1_modality_f = clones(nn.Linear(dim_hidden, dim_mid, bias=True), self.num_feats)
            self.linear2_modality = nn.Linear(dim_mid if fusion_type == 'addition' else dim_mid * 2, 1, bias=False)

        self._init_weights()
        

    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.ModuleList):
                for m in module:
                    nn.init.xavier_normal_(m.weight)
            else:
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
            #print(hidden_state.shape, feats[i].shape)
            _, weight = self.cal_out(
                [self.linear1_temporal_h, self.linear1_temporal_f[i] if self.different_wf else self.linear1_temporal_f],
                #[self.linear1_temporal_h, self.linear1_temporal_f[i]],
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
                #[self.linear1_modality_h, self.linear1_modality_f[i]],
                self.linear2_modality,
                [hidden_state, context[i]]
                )
            weight_modality.append(out)
        weight_modality = F.softmax(torch.cat(weight_modality, dim=1), dim=1)
        final_context = torch.bmm(weight_modality.unsqueeze(1), torch.stack(context, dim=1)).squeeze(1)

        return final_context, weight_modality



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


class LSTM_with_GCC_Cell(torch.nn.Module):
    def extra_repr(self):
        s = '{input_size}, {feats_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        return s.format(**self.__dict__)

    def __init__(self, input_size, feats_size, hidden_size, bias=True):
        super(LSTM_with_GCC_Cell, self).__init__()

        self.input_size = input_size
        self.feats_size = feats_size
        self.hidden_size = hidden_size
        self.bias = bias
        num_gates = 4
        self.num_gates = 4
        self.l = isinstance(self.feats_size, list)
        self.weight_ih = Parameter(torch.Tensor(num_gates * hidden_size, input_size))
        if self.l:
            self.weight_fh = nn.ParameterList([Parameter(torch.Tensor((num_gates-1) * hidden_size, fsize)) for fsize in feats_size])
        else:
            self.weight_fh = Parameter(torch.Tensor((num_gates-1) * hidden_size, feats_size))
            #self.weight_fh = None
        self.weight_hh = Parameter(torch.Tensor(num_gates * hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.Tensor(num_gates * hidden_size))
        if self.l:
            self.bias_fh = nn.ParameterList([Parameter(torch.Tensor((num_gates-1) * hidden_size)) for _ in range(len(feats_size))])
        else:
            self.bias_fh = Parameter(torch.Tensor((num_gates-1) * hidden_size))
            #self.bias_fh = None
        self.bias_hh = Parameter(torch.Tensor(num_gates * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, feats, hidden):
        assert input.size(1) == self.input_size
        if hidden is None:
            hidden = (input.new_zeros(input.size(0), self.hidden_size, requires_grad=False), input.new_zeros(input.size(0), self.hidden_size, requires_grad=False))

        hx, cx = hidden
        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        if self.l:
            tmp = []
            for i in range(len(feats)):
                tmp.append(F.linear(feats[i], self.weight_fh[i], self.bias_fh[i]))

            if len(tmp):
                tmp = torch.stack(tmp, dim=0).sum(0)
                tmpi, tmpf, tmpo = tmp.chunk(3, 1)
                ingate += tmpi
                forgetgate += tmpf
                outgate += tmpo
        else:
            tmp = F.linear(feats, self.weight_fh, self.bias_fh)
            tmpi, tmpf, tmpo = tmp.chunk(3, 1)
            #tmpi, tmpf, tmpo = feats, feats, feats
            ingate += tmpi
            forgetgate += tmpf
            outgate += tmpo

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return (hy, cy), [ingate, forgetgate, outgate]
        



'''
CUDA_VISIBLE_DEVICES=2 python train.py -ss -wc -all -m ami --scope base_Ami --skip_info 0 1 1 -afa mi
'''

class LSTM_Decoder_2stream(nn.Module):
    def __init__(self, opt, embedding=None):
        super(LSTM_Decoder_2stream, self).__init__()

        if 'a' in opt['modality']:
            self.feats_size = [opt['dim_hidden'], opt.get('dim_hidden_a', opt['dim_hidden'])]
        else:
            self.feats_size = [opt['dim_hidden']]


        self.word_size = opt['dim_hidden']

        self.hidden_size = opt["dim_hidden"]
        self.vocab_size = opt["vocab_size"]
        self.max_len = opt['max_len']

        self.embedding = embedding if embedding is not None else nn.Embedding(self.vocab_size, self.word_size)


        lstm_input_size = self.word_size + sum(self.feats_size) + (opt['num_category'] if opt['with_category'] else 0)
        

        self.with_category = opt['with_category']
        self.rnn = nn.LSTMCell(lstm_input_size, self.hidden_size)
        self.forget_bias = opt.get('forget_bias', 0.6)
        self._init_lstm_forget_bias()

        self.att = Attentional_Attention(
                self.hidden_size, 
                self.feats_size, 
                opt.get('att_mid_size', 256),
                different_wf=True
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


class Top_Down_Decoder(nn.Module):
    def __init__(self, opt):
        super(Top_Down_Decoder, self).__init__()
        with_multimodal_attention = opt.get('with_multimodal_attention', False)
        addition = opt.get('addition', False)
        temporal_concat = opt.get('temporal_concat', False)
        gated_sum = opt.get('gated_sum', False)
        bidirectional = opt.get('bidirectional', False)
        use_tag = opt.get('use_tag', False)
        no_tag_emb = opt.get('no_tag_emb', False)
        last_tag = opt.get('last_tag', False)

        use_chain = opt.get('use_chain', False)
        chain_both = opt.get('chain_both', False)

        if (opt['encoder_type'] == 'IPE' or addition or gated_sum or temporal_concat):
            num_modality = 1
        elif use_chain:
            num_modality = 1 if not chain_both else 2
        else:
            num_modality = (len(opt['modality']) - sum(opt['skip_info']))
            

        self.word_size = opt['dim_hidden']
        if opt['encoder_type'] == 'IPE' or with_multimodal_attention or addition or gated_sum or temporal_concat:
            self.feats_size = opt['dim_hidden']  
        else:
            self.feats_size = opt['dim_hidden'] * num_modality

        self.hidden_size = opt["dim_hidden"]
        self.vocab_size = opt["vocab_size"]
        self.max_len = opt['max_len']

        if opt.get('others', False):
            import pickle
            self.embedding = Embedding_Layer(
                    vocab_size = self.vocab_size, 
                    dim_word = 300, 
                    dim_hidden = self.hidden_size, 
                    embedding_weights = pickle.load(open(opt['corpus_pickle'], 'rb'))['glove'], 
                    train_embedding=False
                )
        else:
            #self.embedding = nn.Embedding(self.vocab_size, self.word_size)
            self.embedding = Embedding_Layer_ne2h(self.vocab_size, self.word_size, use_LN=True)#)#, use_LN=True)
        
        if use_tag:
            self.tag_embedding = nn.Embedding(opt["tag_size"], opt['dim_tag'])
            #self.tag_embedding2 = nn.Embedding(opt["tag_size"], opt['dim_tag'])
        if use_tag or last_tag:
            self.tgt_tag_prj = nn.Linear(self.hidden_size, opt['tag_size'], bias=False)
        self.use_tag = use_tag
        self.no_tag_emb = no_tag_emb
        self.last_tag = last_tag

        self.with_category = opt['with_category']

        self.varlstm = opt.get('varlstm', False)
        if self.varlstm:
            self.rnn = VarLSTM(input_size=self.word_size + (opt['dim_tag'] if use_tag else 0), hidden_size=self.hidden_size, 
                batch_first=True, input_dropout=opt['varlstm_id'], hidden_dropout=opt['varlstm_hd'])
            self.rnn2 = VarLSTM(input_size=self.hidden_size + self.feats_size + (opt['num_category'] if opt['with_category'] else 0) + (opt['dim_tag'] if (use_tag and not no_tag_emb) else 0), hidden_size=self.hidden_size,
                batch_first=True, input_dropout=opt['varlstm_id'], hidden_dropout=opt['varlstm_hd'])
        else:
            if opt.get('mylstm', False):
                self.rnn = my_LSTM_Cell(self.word_size + (opt['dim_tag'] if use_tag else 0), self.hidden_size, use_LN=False)
                self.rnn2 = my_LSTM_Cell(self.hidden_size + self.feats_size + (opt['num_category'] if opt['with_category'] else 0) + (opt['dim_tag'] if (use_tag and not no_tag_emb) else 0), self.hidden_size, 
                    use_LN=True)
            else:
                self.rnn = nn.LSTMCell(self.word_size + (opt['dim_tag'] if use_tag else 0), self.hidden_size)
                self.rnn2 = nn.LSTMCell(self.hidden_size + self.feats_size + (opt['num_category'] if opt['with_category'] else 0) + (opt['dim_tag'] if (use_tag and not no_tag_emb) else 0), self.hidden_size)

            self.forget_bias = opt.get('forget_bias', 0.6)
            self._init_lstm_forget_bias()
        
        '''
        self.att = BasicAttention(
                self.hidden_size, 
                [self.feats_size], 
                opt.get('att_mid_size', 256),
            )
        '''
        

        self.att = Attentional_Attention(
                self.hidden_size,# * 2, 
                [opt["dim_hidden"]] * num_modality, 
                opt.get('att_mid_size', 256),
                with_multimodal_attention=with_multimodal_attention,
                bidirectional=bidirectional
            )
        self.dropout = nn.Dropout(opt['decoder_dropout'])
        self.decoder_hidden_init_type = opt.get('decoder_hidden_init_type', 0)

        self.tse = opt.get('task_specific_embedding', False)
        if self.tse:
            self.task_specific_embedding = nn.Embedding(2, self.word_size)

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
        if isinstance(encoder_hidden, list):
            hidden = [(encoder_hidden[0].clone(), encoder_hidden[0].clone()), (encoder_hidden[1].clone(), encoder_hidden[1].clone())] 

        if self.decoder_hidden_init_type == 0:
            hidden = [(encoder_hidden.clone(), encoder_hidden.clone()), (encoder_hidden.clone(), encoder_hidden.clone())]
        elif self.decoder_hidden_init_type == 1:
            hidden = [None, (encoder_hidden.clone(), encoder_hidden.clone())]
        elif self.decoder_hidden_init_type == 2:
            hidden = [None, None]
        elif self.decoder_hidden_init_type == 3:
            hidden = [(encoder_hidden.clone(), encoder_hidden.clone()), None]

        if self.training and self.tse:
            h = []
            for item in hidden:
                if item is not None:
                    h.append((item[0].repeat(2, 1), item[1].repeat(2, 1)))
                else:
                    h.append(None)
            return h
        return hidden



    def forward(self, **kwargs):
        # get the infomation we need from the kwargs
        it, encoder_outputs, category, decoder_hidden, tag = map(
            lambda x: kwargs[x], 
            ["it", "encoder_outputs", "category", "decoder_hidden", "tag"]
        )
        current_words = self.embedding(it)
        if self.use_tag:
            assert tag is not None
            current_tags = self.tag_embedding(tag)
            #print(category.shape, current_tags.shape)

        assert isinstance(decoder_hidden, list)
        assert len(decoder_hidden) == 2

        contents = [current_words]
        
        if self.use_tag:
            contents.append(current_tags)

        input_content = torch.cat(contents, dim=1)
        if self.tse:
            bsz = input_content.size(0)
            if self.training:
                
                input_content = input_content.repeat(2, 1)
                tse_emb = [
                    self.task_specific_embedding(input_content.new(bsz).fill_(0).long()),
                    self.task_specific_embedding(input_content.new(bsz).fill_(1).long()),
                ]
                tse_emb = torch.cat(tse_emb, dim=0)
                input_content += tse_emb

                if isinstance(encoder_outputs, list):
                    encoder_outputs = [item.repeat(2, 1, 1) for item in encoder_outputs]
                else:
                    encoder_outputs = encoder_outputs.repeat(2, 1, 1)
            else:
                tse_emb = self.task_specific_embedding(input_content.new(bsz).fill_(0).long())
                input_content += tse_emb

        if self.varlstm:
            #print(input_content.unsqueeze(1).shape)
            hx = (decoder_hidden[0][0].unsqueeze(0), decoder_hidden[0][1].unsqueeze(0))
            _, hy = self.rnn(input_content.unsqueeze(1), hx)
            decoder_hidden[0] = (hy[0].squeeze(0), hy[1].squeeze(0))
            
        else:
            decoder_hidden[0] = self.rnn(self.dropout(input_content), decoder_hidden[0])

        hidden_state = decoder_hidden[0][0]
        #hidden_state = decoder_hidden[0][0] + decoder_hidden[1][0]
        #hidden_state = torch.cat([decoder_hidden[0][0], decoder_hidden[1][0]], dim=1)

        context, frames_weight = self.att(hidden_state, encoder_outputs if isinstance(encoder_outputs, list) else [encoder_outputs])
        contents = [decoder_hidden[0][0], context]
        if self.with_category:
            contents.append(category)

        pred_next_tag = None
        if self.use_tag and not self.last_tag:
            pred_next_tag = F.log_softmax(self.tgt_tag_prj(self.dropout(hidden_state)))
            if not self.no_tag_emb:
                next_tag = self.tag_embedding(pred_next_tag.argmax(1))
                contents.append(next_tag)

        input_content = torch.cat(contents, dim=1)
        
        #print(input_content.shape)
        if self.varlstm:
            hx = (decoder_hidden[1][0].unsqueeze(0), decoder_hidden[1][1].unsqueeze(0))
            _, hy = self.rnn2(input_content.unsqueeze(1), hx)
            decoder_hidden[1] = (hy[0].squeeze(0), hy[1].squeeze(0))
        else:
            decoder_hidden[1] = self.rnn2(self.dropout(input_content), decoder_hidden[1])

        hidden_state = self.dropout(decoder_hidden[1][0])
        if self.tse and self.training:
            h_for_word, h_for_tag = hidden_state.chunk(2, dim=0)
            if self.last_tag:
                pred_next_tag = F.log_softmax(self.tgt_tag_prj(h_for_tag))
            hidden_state = h_for_word
        else:
            if self.last_tag:
                pred_next_tag = F.log_softmax(self.tgt_tag_prj(hidden_state))

        return {
            'dec_outputs': hidden_state,
            'dec_hidden': decoder_hidden,
            'weights': frames_weight,
            'pred_tag': pred_next_tag
        }


class my_LSTM_Cell(torch.nn.Module):
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        return s.format(**self.__dict__)

    def __init__(self, input_size, hidden_size, bias=True, use_LN=False):
        super(my_LSTM_Cell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        num_gates = 4
        self.weight_ih = Parameter(torch.Tensor(num_gates * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(num_gates * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(num_gates * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(num_gates * hidden_size))

        #self.ln = clones(nn.LayerNorm(hidden_size), num_gates) if use_LN else None
        self.ln = nn.LayerNorm(hidden_size) if use_LN else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):
        assert input.size(1) == self.input_size
        if hidden is None:
            hidden = (input.new_zeros(input.size(0), self.hidden_size, requires_grad=False), input.new_zeros(input.size(0), self.hidden_size, requires_grad=False))

        hx, cx = hidden
        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        if self.ln is not None:
            '''
            ingate = self.ln[0](ingate)
            forgetgate = self.ln[1](forgetgate)
            cellgate = self.ln[2](cellgate)
            outgate = self.ln[3](outgate)
            '''
            ingate = self.ln(ingate)
            forgetgate = self.ln(forgetgate)
            cellgate = self.ln(cellgate)
            outgate = self.ln(outgate)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return (hy, cy)
  