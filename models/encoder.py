from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F


class Input_Embedding_Layer(nn.Module):
    def __init__(self, input_size, hidden_size, skip_info, name=[]):
        super(Input_Embedding_Layer, self).__init__()
        
        self.input_size = input_size
        if len(name) == 0:
            name = [str(i) for i in range(len(self.input_size))]
        else:
            assert len(name) == len(self.input_size)

        self.encoder = []
        for i in range(len(input_size)):
            if skip_info[i] == 0:
                tmp_module = nn.Sequential(
                            *(
                                nn.Linear(input_size[i], hidden_size),
                            )
                        )
                self.add_module("Encoder_%s" % name[i], tmp_module)
                self.encoder.append(tmp_module)
            else:
                self.encoder.append(None)

    def forward(self, **kwargs):
        input_feats = kwargs['input_feats']
        batch_size, seq_len, _ = input_feats[0].shape

        encoder_outputs = []
        for i, feats in enumerate(len(input_feats)):
            if self.encoder[i] is None:
                encoder_output = feats
            else:
                encoder_output = self.encoder[i](feats)
            encoder_outputs.append(encoder_output)

        return encoder_outputs


class Visual_Oriented_Encoder(nn.Module):
    def __init__(self, input_size=[1536, 2048], hidden_size=[512, 1024], opt=None):
        super(Visual_Oriented_Encoder, self).__init__()
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


class MLP(nn.Module):
    def __init__(self, input_size, output_size, with_audio=False):
        super(MLP, self).__init__()
        mul = 5 if with_audio else 4
        self.net = nn.Sequential(
                        *(
                            nn.Linear(input_size, int(mul*output_size)),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(int(mul*output_size), output_size)
                        )
                    )
    def forward(self, input_feats):
        encoder_outputs = self.net(torch.cat(input_feats, dim=2))
        return encoder_outputs, encoder_outputs.mean(1)



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
            num_layers = 1

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

        #return outputs, hiddens
        return torch.stack(outputs, dim=0).mean(0), hiddens


