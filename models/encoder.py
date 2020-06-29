from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class VIP_layer(nn.Module):
    def __init__(self, seq_len, dim_feats, num_class, dropout_ratio, VIP_level=3, weighted_addition=False):
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

        # corresponding dropout
        self.dropout = nn.ModuleList(
                [copy.deepcopy(nn.Dropout(p=dropout_ratio)) for _ in range(VIP_level)]
            )

        # Various-timescale Inference
        self.inference = nn.ModuleList(
                [copy.deepcopy(nn.Linear(dim_feats, num_class)) for _ in range(VIP_level)]
            )

        self.seq_len = seq_len
        self.dim_feats = dim_feats
        self.VIP_level = VIP_level
        self.weights = [(2**n if weighted_addition else 1) for n in range(VIP_level)] 

        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.001)
                nn.init.constant_(m.bias.data, 0)
        '''

    def forward(self, x):
        '''
            input: 
                -- x: [batch_size, seq_len, dim_feats]
            output:
                -- the prediction results: [batch_size, num_class]
        '''   
        x = x.permute(0,2,1).unsqueeze(3).unsqueeze(4) #[batch_size, dim_feats, D=seq_len, H=1, W=1]  
        collections = []
        for n in range(self.VIP_level):
            y = self.pooling[n](x).mean(2) #[batch_size, dim_feats, H=1, W=1]
            y = self.dropout[n](y).view(-1, self.dim_feats) #[batch_size, dim_feats]
            y = self.inference[n](y) #[batch_size, num_class]
            collections.append(self.weights[n] * y)

        results = torch.stack(collections, dim=0).sum(0) # w1 * y1 + w2 * y2 + ...
        return F.sigmoid(results)

class HighWay(nn.Module):
    def __init__(self, hidden_size):
        super(HighWay, self).__init__()
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        y = self.tanh(self.w1(x))
        gate = torch.sigmoid(self.w2(x))
        return gate * x + (1 - gate) * y

class HighWay_GCC(nn.Module):
    """docstring for HighWay_GCC"""
    def __init__(self, hidden_size, num_feats=0, return_gate_info=False):
        super(HighWay_GCC, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.num_feats = num_feats
        if self.num_feats:
            self.wf = nn.ModuleList(
                [copy.deepcopy(nn.Linear(hidden_size, hidden_size)) for _ in range(self.num_feats)]
            )
        self.return_gate_info = return_gate_info

    def forward(self, x, feats=None):
        x = self.dropout(x)

        y = self.tanh(self.w1(x))
        input_ = self.w2(x)
        if feats is not None:
            assert len(feats) == self.num_feats
            feats = [self.dropout(item) for item in feats]
            for i in range(self.num_feats):
                input_ += self.wf[i](feats[i])

        gate = torch.sigmoid(input_)
        res = gate * x + (1 - gate) * y
        if self.return_gate_info:
            return res, res.mean(1), [gate]
        return res, res.mean(1)
        

class Encoder_HighWay(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], auxiliary_pos=[[], [0]], skip_info=[1, 0], return_gate_info=False, use_LSTM=False, opt={}):
        super(Encoder_HighWay, self).__init__()
        assert len(input_size) == len(output_size)
        assert len(input_size) == len(auxiliary_pos)
        assert len(input_size) == len(skip_info)
        assert sum(skip_info) != len(skip_info)

        self.input_size = input_size
        self.output_size = output_size
        self.auxiliary_pos = auxiliary_pos
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

                tmp_module = HighWay_GCC(input_size[i], len(auxiliary_size), return_gate_info=return_gate_info)
                self.encoder.append(tmp_module)
                self.add_module('Encoder%s'%name[i], tmp_module)

    def will_return_gate_info(self):
        return self.return_gate_info

    def forward(self, input_feats):
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

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
                eo, eh, g = self.encoder[i](input_feats[i], auxiliary_feats)
                gate.append(g)
            else:
                eo, eh = self.encoder[i](input_feats[i], auxiliary_feats)
            outputs.append(eo)
            hiddens.append(eh)

        if self.return_gate_info:
            return outputs, hiddens, gate
        else:
            return outputs, hiddens


class HighWay_IEL(nn.Module):
    def __init__(self, input_size, hidden_size, name=[], dropout=0.5):
        super(HighWay_IEL, self).__init__()
        if isinstance(hidden_size, list):
            hidden_size = hidden_size[0]
        
        self.input_size = input_size
        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        self.encoder = []
        for i in range(len(input_size)):
            tmp_module = nn.Sequential(
                        *(
                            nn.Linear(input_size[i], hidden_size),
                            HighWay(hidden_size),
                            nn.Dropout(dropout),
                        )
                    )
            self.add_module("Encoder_%s" % name[i], tmp_module)
            self.encoder.append(tmp_module)

    def forward(self, input_feats):
        #batch_size, seq_len, _ = input_feats[0].shape

        encoder_ouputs = []
        encoder_hiddens = []
        for i in range(len(input_feats)):
            input_ = input_feats[i] if len(input_feats[i].shape) > 2 else input_feats[i].unsqueeze(1)
            encoder_ouput = self.encoder[i](input_)
            encoder_ouputs.append(encoder_ouput)

        for item in encoder_ouputs:
            encoder_hiddens.append(item.mean(1))

        return encoder_ouputs, encoder_hiddens
        #return torch.cat(encoder_ouputs, dim=1), torch.stack(encoder_hiddens, dim=0).mean(0)


class LEL(nn.Module):
    def __init__(self, input_size, hidden_size, name=[], dropout=0.5):
        super(LEL, self).__init__()
        if isinstance(hidden_size, list):
            hidden_size = hidden_size[0]
        
        self.input_size = input_size
        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        self.encoder = []
        for i in range(len(input_size)):
            tmp_module = nn.Sequential(
                        *(
                            nn.Linear(input_size[i], hidden_size),
                            nn.Dropout(dropout),
                        )
                    )
            self.add_module("Encoder_%s" % name[i], tmp_module)
            self.encoder.append(tmp_module)

    def forward(self, input_feats):
        encoder_ouputs = []
        encoder_hiddens = []
        for i in range(len(input_feats)):
            input_ = input_feats[i] if len(input_feats[i].shape) > 2 else input_feats[i].unsqueeze(1)
            encoder_ouput = self.encoder[i](input_)
            encoder_ouputs.append(encoder_ouput)

        for item in encoder_ouputs:
            encoder_hiddens.append(item.mean(1))

        return encoder_ouputs, encoder_hiddens


class Input_Embedding_Layer(nn.Module):
    def __init__(self, input_size, hidden_size, name=[]):
        super(Input_Embedding_Layer, self).__init__()
        
        self.input_size = input_size
        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        self.encoder = []
        for i in range(len(input_size)):
            tmp_module = nn.Sequential(
                        *(
                            nn.Linear(input_size[i], hidden_size),
                        )
                    )
            self.add_module("Encoder_%s" % name[i], tmp_module)
            self.encoder.append(tmp_module)

    def forward(self, **kwargs):
        input_feats = kwargs['input_feats']
        batch_size, seq_len, _ = input_feats[0].shape

        encoder_outputs = []
        for i in range(len(input_feats)):
            input_ = input_feats[i]
            encoder_output = self.encoder[i](input_)
            encoder_outputs.append(encoder_output)

        return encoder_outputs

class Semantics_Enhanced_IEL(nn.Module):
    def __init__(self, input_size, semantics_size, nf, name, multiply=False):
        super(Semantics_Enhanced_IEL, self).__init__()
        assert len(name) == len(input_size)
        self.f2e = []
        for i in range(len(input_size)):
            tmp_module = nn.Linear(input_size[i], nf)
            self.add_module("%s2e" % name[i], tmp_module)
            self.f2e.append(tmp_module)

        self.s2e = nn.Linear(semantics_size, nf)
        self.dropout = nn.Dropout(0.5)
        self.multiply = multiply

    def forward(self, input_feats, semantics):
        #se = self.s2e(semantics)
        se = self.s2e(self.dropout(semantics))
        encoder_outputs = []
        for i in range(len(input_feats)):
            #fe = self.f2e[i](input_feats[i])
            fe = self.f2e[i](self.dropout(input_feats[i]))
            res = (se * fe) if self.multiply else (se + fe)
            encoder_outputs.append(res)

        return encoder_outputs



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        #print(scores.shape, mask.shape)
        scores = scores.masked_fill(mask == 0, -1e9)
    #print(scores.shape ,mask.shape)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)       # h
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        #return self.linears[-1](x)
        return x



class Encoder_Baseline(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], auxiliary_pos=[[], [0]], skip_info=[1, 0], return_gate_info=False, use_LSTM=False, opt={}):
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
        bidirectional = opt.get('bidirectional', False)
        for i in range(len(input_size)):
            if skip_info[i]:
                self.encoder.append(None)
            else:
                auxiliary_size = []
                for pos in self.auxiliary_pos[i]:
                    auxiliary_size.append(input_size[pos])
                if bidirectional:
                    rnn = Bi_GRU_with_GCC
                else:
                    rnn = GRU_with_GCC if not use_LSTM else LSTM_with_GCC
                tmp_module = rnn(input_size[i], auxiliary_size, output_size[i], return_gate_info=return_gate_info, attention=opt.get('gate_attention', False),
                        addition=opt.get('gcc_addition', False), init=opt.get('eh_init', False)
                    )
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

        #outputs = [outputs[-1]]

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

    def __init__(self, input_size, feats_size, hidden_size, bias=True, return_gate_info=False, attention=False, addition=False, init=False):
        super(GRU_with_GCC, self).__init__()
        assert isinstance(feats_size, list)
        self.input_size = input_size
        self.feats_size = feats_size
        self.hidden_size = hidden_size
        self.bias = bias
        num_gates = 3
        self.num_gates = 3
        self.addition = addition

        self.weight_ih = Parameter(torch.Tensor(num_gates * hidden_size, input_size))
        if self.addition:
            self.weight_fh = None
        else:
            self.weight_fh = nn.ParameterList([Parameter(torch.Tensor((num_gates-1) * hidden_size, fsize)) for fsize in feats_size])
        self.weight_hh = Parameter(torch.Tensor(num_gates * hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.Tensor(num_gates * hidden_size))
        if self.addition:
            self.bias_fh = None
        else:
            self.bias_fh = nn.ParameterList([Parameter(torch.Tensor((num_gates-1) * hidden_size)) for _ in range(len(feats_size))])
        #self.bias_fh = None
        self.bias_hh = Parameter(torch.Tensor(num_gates * hidden_size))

        self.return_gate_info = return_gate_info

        self.reset_parameters()
        self.attention = attention
        if len(feats_size) and attention:
            self.att = MultiHeadedAttention(h=1, d_model=hidden_size)

        self.init = init
        if init:
            self.feats2hidden = nn.Sequential(
                        *(
                            nn.Linear(input_size, hidden_size),
                            nn.Tanh()
                        )
                    )

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

        if self.addition:
            feats.append(input)
            input = torch.stack(feats, dim=0).sum(0)

        gi = F.linear(input, weight_ih, bias_ih)
        gh = F.linear(hx, weight_hh, bias_hh)

        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        if not self.addition:
            if self.attention:
                gf = [[], []]
                for i in range(len(feats)):
                    #gf.append(F.linear(feats[i], weight_fh[i], bias_fh[i]))
                    t = F.linear(feats[i], weight_fh[i], bias_fh[i])
                    t1, t2 = t.chunk(2, 1)
                    gf[0].append(t1)
                    gf[1].append(t2)

                if len(gf[0]): 
                    query = hx.unsqueeze(1)
                    gf[0].append(i_r)
                    gf[1].append(i_i)
                    r = torch.stack(gf[0], dim=1)
                    rres = self.att(query=query, key=r, value=r).squeeze(1)
                    #print(rres.shape, h_r.shape)
                    resetgate = F.sigmoid(rres + h_r)

                    tmp = torch.stack(gf[1], dim=1)
                    tmpres = self.att(query=query, key=tmp, value=tmp).squeeze(1)
                    inputgate = F.sigmoid(tmpres + h_i)
                else:
                    resetgate = F.sigmoid(i_r + h_r)
                    inputgate = F.sigmoid(i_i + h_i)
            else:
                gf = []
                for i in range(len(feats)):
                    gf.append(F.linear(feats[i], weight_fh[i], bias_fh[i]))
                    #gf.append(F.linear(feats[i], weight_fh[i]))
                if len(gf):
                    gf = torch.stack(gf, dim=0).sum(0)
                    #gf = torch.stack(gf, dim=0).mean(0)
                    f_r, f_i = gf.chunk(2, 1)
                    resetgate = F.sigmoid(i_r + h_r + f_r)
                    inputgate = F.sigmoid(i_i + h_i + f_i)
                else:
                    resetgate = F.sigmoid(i_r + h_r)
                    inputgate = F.sigmoid(i_i + h_i)
        else:
            resetgate = F.sigmoid(i_r + h_r)
            inputgate = F.sigmoid(i_i + h_i)
        
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)
        
        return hy, [resetgate, inputgate]

    def forward(self, input_feats, auxiliary_feats, hx=None, reverse=False):
        batch_size, seq_len, _ = input_feats.shape
        output = []
        gate = [[] for _ in range(self.num_gates-1)]


        if self.init:
            hx = self.feats2hidden(input_feats.mean(1))

        #afeats = [item.mean(1).unsqueeze(1).repeat(1, 8, 1) for item in auxiliary_feats]

        for i in range(seq_len):
            input = input_feats[:, seq_len - 1 - i if reverse else i, :]
            feats = [item[:, seq_len - 1 - i if reverse else i, :] for item in auxiliary_feats]
            hy, g = self.forward_each_timestep(input, feats, hx, 
                self.weight_ih, self.weight_fh, self.weight_hh, 
                self.bias_ih, self.bias_fh, self.bias_hh)
            output.append(hy.clone())
            if self.return_gate_info:
                for j, item in enumerate(g):
                    gate[j].append(item)
            hx = hy

        if reverse:
            output = output[::-1]
        output = torch.stack(output, dim=1)
        #print(self.weight_ih.max(), self.weight_ih.min())
        if self.return_gate_info:
            for j in range(self.num_gates-1):
                gate[j] = torch.stack(gate[j], dim=1)
            return output, hx, gate
        return output, hx

class Bi_GRU_with_GCC(torch.nn.Module):
    def __init__(self, input_size, feats_size, hidden_size, bias=True, return_gate_info=False):
        super(Bi_GRU_with_GCC, self).__init__()
        self.forward_rnn = GRU_with_GCC(input_size, feats_size, hidden_size, bias, return_gate_info)
        self.backward_rnn = GRU_with_GCC(input_size, feats_size, hidden_size, bias, return_gate_info)
        self.return_gate_info = return_gate_info

    def forward(self, input_feats, auxiliary_feats, hx=None):
        if self.return_gate_info:
            fo, fh, r1 = self.forward_rnn(input_feats, auxiliary_feats, hx=None)
            bo, bh, r2 = self.backward_rnn(input_feats, auxiliary_feats, hx=None, reverse=True)
            return torch.cat([fo, bo], dim=2), torch.stack([fh, bh], dim=0).mean(0), [r1, r2]
        else:
            fo, fh = self.forward_rnn(input_feats, auxiliary_feats, hx=None)
            bo, bh = self.backward_rnn(input_feats, auxiliary_feats, hx=None, reverse=True)
            return torch.cat([fo, bo], dim=2), torch.stack([fh, bh], dim=0).mean(0)


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



class Encoder_Baseline_TwoStream(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], auxiliary_pos=[[], [0]], skip_info=[1, 0], return_gate_info=False, use_LSTM=False):
        super(Encoder_Baseline_TwoStream, self).__init__()
        assert len(input_size) == len(output_size)
        assert len(input_size) == len(auxiliary_pos)
        assert len(input_size) == len(skip_info)
        assert sum(skip_info) != len(skip_info)

        self.input_size = input_size
        self.output_size = output_size
        self.auxiliary_pos = auxiliary_pos
        self.dropout = nn.Dropout(0.5)
        self.return_gate_info = return_gate_info

        #if len(name) == 0:
        #    name = [str(i) for i in range(len(self.input_size))]
        #else:
        assert len(name) == len(self.input_size)
        self.visual = []
        self.audio = []
        for i, char in enumerate(name):
            if char in ['M', 'I']:
                self.visual.append(i)
            else:
                self.audio.append(i)
        print('Visual: %s' % ' '.join([str(i) for i in self.visual]))
        print('Audio: %s' % ' '.join([str(i) for i in self.audio]))


        self.weights = []
        self.bias = []
        self.linear_list = []
        for i in range(len(input_size)):
            ng = 3 if not skip_info[i] else 2
            self.register_parameter('shared_weight%d'%i, Parameter(torch.Tensor(ng * output_size[i], input_size[i])))
            self.register_parameter('shared_bias%d'%i, Parameter(torch.Tensor(ng * output_size[i])))
            
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

        visual = [[], []]
        audio = [[], []]

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
            
            if i in self.visual:
                visual[0].append(eo)
                visual[1].append(eh)
            else:
                audio[0].append(eo)
                audio[1].append(eh)


        
        outputs = self.decide_output(visual[0], audio[0])
        hidden = self.decide_hidden(visual[1], audio[1])

        if self.return_gate_info:
            return outputs, hidden, gate
        else:
            return outputs, hidden

    def decide_output(self, visual_feats, audio_feats):
        outputs = []
        if len(visual_feats):
            visual_output = torch.stack(visual_feats, dim=0).mean(0)
            outputs.append(visual_output)
        if len(audio_feats):
            audio_output = torch.stack(audio_feats, dim=0).mean(0)
            outputs.append(audio_output)
        

        return outputs

    def decide_hidden(self, visual_feats, audio_feats):
        visual_hidden = torch.stack(visual_feats, dim=0).mean(0)
        #audio_hidden = 
        return visual_hidden


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
        #for weight in self.parameters():
        for name, weight in self.named_parameters():
            if 'weight_b' in name:
                weight.data.uniform_(1-stdv, 1+stdv)
            else:    
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

class GRU_with_GCC_with_shared_params(torch.nn.Module):
    def extra_repr(self):
        s = 'hidden={hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        return s.format(**self.__dict__)

    def __init__(self, input_size, hidden_size, bias=True, return_gate_info=False, attention=False, init=False):
        super(GRU_with_GCC_with_shared_params, self).__init__()
        self.hidden_size = hidden_size
        self.bias = bias

        self.num_gates = 3
        self.weight_hh = Parameter(torch.Tensor(self.num_gates * hidden_size, hidden_size))
        self.bias_hh = Parameter(torch.Tensor(self.num_gates * hidden_size))
        self.return_gate_info = return_gate_info
        self.reset_parameters()
        
        self.attention = attention
        if attention:
            self.att = MultiHeadedAttention(h=1, d_model=hidden_size)

        self.init = init
        if init:
            self.feats2hidden = nn.Sequential(
                        *(
                            nn.Linear(input_size, hidden_size),
                            nn.Tanh()
                        )
                    )

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

            num = tmp.shape[1] // self.hidden_size
            if num == self.num_gates:
                tmpr, tmpi, _ = tmp.chunk(3, 1)
            elif num > self.num_gates:
                if num % 3 == 0:
                    tmp = F.avg_pool1d(tmp.unsqueeze(1), kernel_size=num//3).squeeze(1)
                    tmpr, tmpi, _ = tmp.chunk(3, 1)
                else:
                    tmp = F.avg_pool1d(tmp.unsqueeze(1), kernel_size=num//2).squeeze(1)
                    tmpr, tmpi = tmp.chunk(2, 1)
            else:
                tmpr, tmpi = tmp.chunk(2, 1)
            fr.append(tmpr)
            fi.append(tmpi)

        if len(fr): 
            '''
            query = hx.unsqueeze(1)
            fr.append(i_r)
            fi.append(i_i)
            r = torch.stack(fr, dim=1)
            rres = self.att(query=query, key=r, value=r).squeeze(1)
            #print(rres.shape, h_r.shape)
            resetgate = F.sigmoid(rres + h_r)


            tmp = torch.stack(fi, dim=1)
            tmpres = self.att(query=query, key=tmp, value=tmp).squeeze(1)
            inputgate = F.sigmoid(tmpres + h_i)
            '''
            f_r = torch.stack(fr, dim=0).sum(0)
            f_i = torch.stack(fi, dim=0).sum(0)
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
        if self.init:
            hx = self.feats2hidden(input_feats.mean(1))

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


'''
class Encoder_Baseline(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], auxiliary_pos=[[], [0]], skip_info=[1, 0], return_gate_info=False, use_LSTM=False, opt={}):
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
                tmp_module = rnn(input_size[i], output_size[i], return_gate_info=return_gate_info, init=opt.get('eh_init', False))
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
'''


'''
CUDA_VISIBLE_DEVICES=2 python train.py -ss -wc -all -m ami --scope base_Ami --skip_info 0 1 1 -afa mi
'''

class Progressive_Encoder(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], opt={}, return_gate_info=False):
        super(Progressive_Encoder, self).__init__()
        assert (opt['modality'].lower() == 'ami') or (opt['modality'].lower() == 'mi')
        self.num_modality = len(opt['modality'])

        self.input_size = input_size
        self.output_size = output_size
        self.dropout = nn.Dropout(0.5)
        self.return_gate_info = return_gate_info
        rnn = GRU_with_GCC
        if self.num_modality == 3:
            self.encoderM = rnn(input_size[1], [input_size[0]], output_size[0], return_gate_info=return_gate_info, attention=opt.get('gate_attention', False))
            self.encoderI_with_auxiliaryM = rnn(input_size[2], [output_size[0]], output_size[1], return_gate_info=return_gate_info, attention=opt.get('gate_attention', False))
        else:
            self.encoderM = rnn(input_size[0], [], output_size[0], return_gate_info=return_gate_info, attention=opt.get('gate_attention', False))
            self.encoderI_with_auxiliaryM = rnn(input_size[1], [output_size[0]], output_size[1], return_gate_info=return_gate_info, attention=opt.get('gate_attention', False))
        self.chain_both = opt.get('chain_both', False)

    def will_return_gate_info(self):
        return self.return_gate_info

    def forward(self, input_feats):
        batch_size, seq_len, _ = input_feats[0].shape
        for i in range(1, len(input_feats)):
            assert input_feats[i].shape[1] == seq_len

        input_feats = [self.dropout(item) for item in input_feats]

        if self.return_gate_info:  
            eo1, eh1, g1 = self.encoderM(input_feats[1 if self.num_modality == 3 else 0], [input_feats[0]] if self.num_modality == 3 else [], hx=None)
            eo2, eh2, g2 = self.encoderI_with_auxiliaryM(input_feats[2 if self.num_modality == 3 else 1], [eo1], hx=None)
            gate = [g1, g2] if self.chain_both else [g2]
        else:
            eo1, eh1 = self.encoderM(input_feats[1 if self.num_modality == 3 else 0], [input_feats[0]] if self.num_modality == 3 else [], hx=None)
            eo2, eh2 = self.encoderI_with_auxiliaryM(input_feats[2 if self.num_modality == 3 else 1], [eo1], hx=None)

        if self.chain_both:
            outputs = [eo1, eo2]
            hiddens = [eh1, eh2]
        else:
            outputs = [eo2]
            hiddens = [eh2]

        if self.return_gate_info:
            return outputs, hiddens, gate
        else:
            return outputs, hiddens


class SVD_Encoder(nn.Module):
    def __init__(self, input_size=[1536, 2048], output_size=[512, 512], name=[], auxiliary_pos=[[], [0]], skip_info=[1, 0], return_gate_info=False, use_LSTM=False, num_factor=512):
        super(SVD_Encoder, self).__init__()
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
                #print(w.max())
                #print(w.min())
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