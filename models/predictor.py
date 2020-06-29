import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import models.Constants as Constants
import torch.nn.functional as F

class Length_Predictor(nn.Module):
    def __init__(self, opt):
        super(Length_Predictor, self).__init__()
        self.use_kl = opt.get('use_kl', False)
        if self.use_kl:
            self.net = nn.Sequential(
                        nn.Linear(opt['dim_hidden'], opt['dim_hidden']),
                        nn.ReLU(),
                        nn.Dropout(opt['hidden_dropout_prob']),
                        nn.Linear(opt['dim_hidden'], opt['max_len']),
                    )
        else:
            self.net = nn.Sequential(
                        nn.Linear(opt['dim_hidden'], opt['dim_hidden']),
                        nn.ReLU(),
                        nn.Dropout(opt['hidden_dropout_prob']),
                        nn.Linear(opt['dim_hidden'], opt['max_len']),
                        nn.ReLU()
                    )
    def forward(self, encoder_outputs):
        if isinstance(encoder_outputs, list):
            assert len(encoder_outputs) == 1
            encoder_outputs = encoder_outputs[0]
        assert len(encoder_outputs.shape) == 3

        out = self.net(encoder_outputs.mean(1))
        if self.use_kl:
            return F.log_softmax(out, dim=-1)
        else:
            return out

class Attribute_Predictor(nn.Module):
    def __init__(self, opt, dim_out):
        super(Attribute_Predictor, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(opt['dim_hidden'], opt['dim_hidden']),
                    nn.ReLU(),
                    nn.Dropout(opt['hidden_dropout_prob']),
                    nn.Linear(opt['dim_hidden'], dim_out),
                    nn.Sigmoid()
                )

    def forward(self, encoder_outputs):
        if isinstance(encoder_outputs, list):
            encoder_outputs = torch.stack(encoder_outputs, dim=0).sum(0)
            #assert len(encoder_outputs) == 1
            #encoder_outputs = encoder_outputs[0]
        assert len(encoder_outputs.shape) == 3

        out = self.net(encoder_outputs.mean(1))
        return out

class Auxiliary_Task_Predictor(nn.Module):
    """docstring for auxiliary_task_predictor"""
    def __init__(self, opt):
        super(Auxiliary_Task_Predictor, self).__init__()
        check_list = ['obj', 'length', 'attr']
        task_mapping = {
            'obj': ('predictor_obj', Attribute_Predictor(opt, opt['dim_object'])),
            'length': ('predictor_length', Length_Predictor(opt)),
            'attr': ('predictor_attr', Attribute_Predictor(opt, opt.get('dim_t', 1000)))
        }

        self.predictor = []
        self.results_names = []

        for item in check_list:
            if item in opt['crit']:
                name, module = task_mapping[item]
                self.predictor.append(module)
                self.add_module(name, module)

                self.results_names.append(Constants.mapping[item][0])

    def forward(self, encoder_outputs):
        results = {}
        for name, pred in zip(self.results_names, self.predictor):
            results[name] = pred(encoder_outputs)
        return results

        