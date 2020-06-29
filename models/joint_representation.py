import torch
import torch.nn as nn

class Gated_Sum(nn.Module):
    def __init__(self, opt):
        super(Gated_Sum, self).__init__()
        hidden_size = opt['dim_hidden']
        nf = opt.get('num_factor', 512)

        self.hidden_size = hidden_size

        self.num_feats = len(opt['modality']) - sum(opt['skip_info'])
        
        #self.emb_weight = Parameter(torch.Tensor(self.num_feats * hidden_size, hidden_size))
        #self.emb_bias = Parameter(torch.Tensor(self.num_feats * hidden_size))

        self.weight_a = Parameter(torch.Tensor(self.num_feats * hidden_size, nf))
        self.weight_b = Parameter(torch.Tensor(nf, self.num_feats))
        self.weight_c = Parameter(torch.Tensor(nf, hidden_size))

        self.bias = Parameter(torch.Tensor(self.num_feats * hidden_size))
        self.dropout = nn.Dropout(0.5)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_gated_result(self, weight, bias, feats, index):
        assert len(feats) == self.num_feats
        #assert len(feats.shape)

        #ew = self.emb_weight.chunk(self.num_feats, 0)
        #eb = self.emb_bias.chunk(self.num_feats, 0)

        w = weight.chunk(self.num_feats, 0)
        b = bias.chunk(self.num_feats, 0)

        res = []
        for i in range(self.num_feats):
            #if i == index:
            #    emb = F.linear(feats[i], ew[i], eb[i])
            res.append(F.linear(self.dropout(feats[i]), w[i], b[i]))
            #res.append(F.linear(feats[i], w[i], b[i]))

        gated_result = F.sigmoid(torch.stack(res, 0).sum(0)) * feats[index]
        #gated_result = F.sigmoid(torch.stack(res, 0).sum(0)) * emb
        return gated_result



    def forward(self, encoder_outputs):
        bsz, seq_len, _ = encoder_outputs[0].shape
        feats = [item.contiguous().view(bsz * seq_len, -1) for item in encoder_outputs]
        #feats = [self.dropout(item.contiguous().view(bsz * seq_len, -1)) for item in encoder_outputs]
        

        gated_results = []
        for i in range(self.num_feats):
            tag = torch.zeros(self.num_feats, 1).to(feats[0].device)
            tag[i] = 1

            #query = feats[i].mean(0).unsqueeze(0).repeat(self.num_feats, 1)    # [3, dim] 
            #key = torch.stack(feats, 1).mean(0)                                # [3, dim]
            #tag = F.cosine_similarity(query, key).unsqueeze(1)

            weight_mid = torch.mm(self.weight_b, tag)
            weight_mid = torch.diag(weight_mid.squeeze(1))
            weight = torch.mm(torch.mm(self.weight_a, weight_mid), self.weight_c)

            gated_results.append(self.get_gated_result(weight, self.bias, feats, i))

        gated_results = torch.stack(gated_results, 0).sum(0)
        gated_results = gated_results.contiguous().view(bsz, seq_len, self.hidden_size)
        
        return gated_results


class Joint_Representaion_Learner(nn.Module):
    def __init__(self, feats_size, opt):
        super(Joint_Representaion_Learner, self).__init__()

        self.encoder_type = opt['encoder_type']
        self.decoder_type = opt['decoder_type']
        self.addition = opt.get('addition', False)
        self.temporal_concat = opt.get('temporal_concat', False)
        self.opt = opt

        self.att = None
        if opt['multi_scale_context_attention']:
            from models.rnn import Multi_Scale_Context_Attention
            self.att = Multi_Scale_Context_Attention(opt)

        if opt.get('gated_sum', False):
            self.att = Gated_Sum(opt)

        self.bn_list = []
        if not opt['no_encoder_bn']:
            if self.addition:
                feats_size = [feats_size[0]]
            print(self.addition)
            print(feats_size)
            for i, item in enumerate(feats_size):
                tmp_module = nn.BatchNorm1d(item)
                self.bn_list.append(tmp_module)
                self.add_module("bn%d"%(i), tmp_module)

    def forward(self, encoder_outputs, encoder_hiddens):
        if self.decoder_type != 'ENSEMBLE' and self.encoder_type == 'GRU' and not self.opt.get('two_stream', False) or self.encoder_type == 'IEL':
            if isinstance(encoder_hiddens[0], tuple):
                hx = []
                cx = []
                for h in encoder_hiddens:
                    hx.append(h[0])
                    cx.append(h[1])
                encoder_hiddens = (torch.stack(hx, dim=0).mean(0), torch.stack(cx, dim=0).mean(0))
            else:
                encoder_hiddens = torch.stack(encoder_hiddens, dim=0).mean(0)


        if self.att is not None:
            encoder_outputs = self.att(encoder_outputs)

        if self.addition:
            assert isinstance(encoder_outputs, list)
            encoder_outputs = torch.stack(encoder_outputs, dim=0).mean(0)
            #encoder_outputs = torch.stack(encoder_outputs, dim=0).max(0)[0]
            
        encoder_outputs = encoder_outputs if isinstance(encoder_outputs, list) else [encoder_outputs]

        if len(self.bn_list):
            assert len(encoder_outputs) == len(self.bn_list)
            
            for i in range(len(encoder_outputs)):
                batch_size, seq_len, _ = encoder_outputs[i].shape
                encoder_outputs[i] = self.bn_list[i](encoder_outputs[i].contiguous().view(batch_size * seq_len, -1)).view(batch_size, seq_len, -1)

        if self.temporal_concat:
            assert isinstance(encoder_outputs, list)
            encoder_outputs = torch.cat(encoder_outputs, dim=1)
            #print(encoder_outputs.shape)

        return encoder_outputs, encoder_hiddens