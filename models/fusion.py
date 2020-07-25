import torch
import torch.nn as nn

class Joint_Representaion_Learner(nn.Module):
    def __init__(self, feats_size, fusion_type, with_bn=True):
        super(Joint_Representaion_Learner, self).__init__()
        self.feats_size = feats_size if isinstance(feats_size, list) else [feats_size]
        self.fusion_type = fusion_type
        self.with_bn = with_bn
        self._check_if_parameters_valid()

        if self.fusion_type == 'mean':
            self.feats_size = [self.feats_size[0]]
        if self.with_bn:
            self.bn_list = nn.ModuleList([nn.BatchNorm1d(size) for size in self.feats_size])

    def forward(self, encoder_outputs, encoder_hiddens):
        encoder_outputs = self._process_encoder_outputs(encoder_outputs)
        encoder_hiddens = self._process_encoder_hiddens(encoder_hiddens)
        return encoder_outputs, encoder_hiddens


    def _process_encoder_outputs(self, encoder_outputs):
        encoder_outputs = encoder_outputs if isinstance(encoder_outputs, list) else [encoder_outputs]

        if self.fusion_type == 'mean':
            encoder_outputs = [torch.stack(encoder_outputs, dim=0).mean(0)]

        if self.with_bn:
            assert len(encoder_outputs) == len(self.bn_list)
            for i in range(len(encoder_outputs)):
                batch_size, seq_len, _ = encoder_outputs[i].shape
                encoder_outputs[i] = encoder_outputs[i].contiguous().view(batch_size * seq_len, -1)
                encoder_outputs[i] = self.bn_list[i](encoder_outputs[i])
                encoder_outputs[i] = encoder_outputs[i].view(batch_size, seq_len, -1)

        if self.fusion_type == 'temporal_concat':
            encoder_outputs = [torch.cat(encoder_outputs, dim=1)]

        return encoder_outputs

    def _process_encoder_hiddens(self, encoder_hiddens):
        encoder_hiddens = encoder_hiddens if isinstance(encoder_hiddens, list) else [encoder_hiddens]
        if isinstance(encoder_hiddens[0], tuple):
            # LSTM
            hx = []
            cx = []
            for h in encoder_hiddens:
                hx.append(h[0])
                cx.append(h[1])
            encoder_hiddens = (torch.stack(hx, dim=0).mean(0), torch.stack(cx, dim=0).mean(0))
        else:
            # GRU or others
            encoder_hiddens = torch.stack(encoder_hiddens, dim=0).mean(0)
        return encoder_hiddens

    def _check_if_parameters_valid(self):
        assert self.fusion_type in ['mean', 'concat', 'temporal_concat']
        if self.fusion_type == 'temporal_concat':
            for i in range(len(self.feats_size) - 1):
                assert self.feats_size[i] == self.feats_size[i+1]

    def _check_if_inputs_valid(self, inputs):
        pass