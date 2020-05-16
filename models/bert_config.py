import copy
import json
import os
from io import open

class PretrainedConfig(object):

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.torchscript = kwargs.pop('torchscript', False)
        self.use_bfloat16 = kwargs.pop('use_bfloat16', False)
        self.pruned_heads = kwargs.pop('pruned_heads', {})

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            setattr(config, key, value)
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

class BertConfig(PretrainedConfig):
    def __init__(self,
                 vocab_size=10546,
                 dim_hidden=512,
                 num_hidden_layers_encoder=2,
                 num_hidden_layers_decoder=1,
                 num_attention_heads=8,
                 intermediate_size=2048,
                 hidden_act="gelu_new",
                 hidden_dropout_prob=0.5,
                 attention_probs_dropout_prob=0.0,
                 max_len=30,
                 layer_norm_eps=1e-12,
                 feat_act="gelu_new",
                 num_category=20,
                 with_category=True,
                 watch=0,
                 residual=[1, 1, 1, 1],
                 pos_attention=False,
                 enhance_input=0,
                 with_layernorm=False,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dim_hidden = dim_hidden
        self.num_hidden_layers_encoder = num_hidden_layers_encoder
        self.num_hidden_layers_decoder = num_hidden_layers_decoder
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_len = max_len
        self.layer_norm_eps = layer_norm_eps


        self.feat_act = feat_act
        self.num_category = num_category
        self.with_category = with_category
        self.watch = watch
        self.residual = residual
        self.pos_attention = pos_attention
        self.enhance_input = enhance_input #0: nothing; 1: resampling; 2: mean
        self.with_layernorm = with_layernorm
