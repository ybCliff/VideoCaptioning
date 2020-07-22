import torch
import torch.nn as nn
from .encoder import Input_Embedding_Layer, Visual_Oriented_Encoder, MLP, Encoder_Baseline
from .fusion import Joint_Representaion_Learner
from .decoder import LSTM_Decoder, Top_Down_Decoder
from .seq2seq import Seq2Seq

def get_preEncoder(opt, input_size):
    preEncoder = None
    output_size = input_size.copy()

    if opt.get('use_preEncoder', False):
        pem = opt.get('preEncoder_modality', '')
        if pem:
            skip_info = [1] * len(opt['modality'])
            for char in pem:
                pos = opt['modality'].index(char)
                skip_info[pos] = 0
                output_size[pos] = opt['dim_hidden']
        else:
            output_size = [opt.get('dim_iel', opt['dim_hidden'])] * len(input_size)
            skip_info = [0] * len(opt['modality'])
        preEncoder = Input_Embedding_Layer(
                input_size=input_size,
                hidden_size=opt.get('dim_iel', opt['dim_hidden']), 
                skip_info=skip_info,
                name=opt['modality'].upper()
            )

    return preEncoder, output_size

def get_encoder(opt, input_size, mapping, modality):
    hidden_size = [opt['dim_hidden']] * len(modality)
    if opt['encoder_type'] == 'IPE':
        if opt.get('MLP', False):
            
            encoder = MLP(sum(input_size), opt['dim_hidden'], 'a' in modality)
        elif opt.get('MSLSTM', False):
            encoder = Encoder_Baseline(input_size=input_size, output_size=hidden_size, name=modality.upper(), encoder_type='mslstm')
        else:
            encoder = Visual_Oriented_Encoder(input_size = input_size, hidden_size = hidden_size, opt = opt)

    return encoder

def get_joint_representation_learner(opt):
    modality = opt['modality'].lower()
    if opt['encoder_type'] == 'GRU':
        if opt.get('use_chain', False):
            feats_size = [opt['dim_hidden'], opt['dim_hidden']] if opt.get('chain_both') else [opt['dim_hidden']]
        elif (opt['multi_scale_context_attention'] and not opt.get('query_all', False)) or opt.get('addition', False) or opt.get('gated_sum', False) or opt.get('temporal_concat', False):
            feats_size = [opt['dim_hidden']]
        elif opt.get('two_stream', False):
            if 'a' in opt['modality']:
                feats_size = [opt['dim_hidden'], opt.get('dim_hidden_a', opt['dim_hidden'])]
            else:
                feats_size = [opt['dim_hidden']]
        else:
            feats_size = [opt['dim_hidden'] * (2 if opt.get('bidirectional', False) else 1)] * (len(modality) - sum(opt['skip_info']))
    elif opt['encoder_type'] in ['IEL', 'LEL']:
        feats_size = [opt['dim_hidden']] * len(modality)
    else:
        feats_size = [opt['dim_hidden']]
    
    return Joint_Representaion_Learner(feats_size, opt)

def get_decoder(opt):
    if opt['decoder_type'] == 'LSTM':
        if opt.get('top_down', False):
            decoder = Top_Down_Decoder(opt)
        else:
            decoder = LSTM_Decoder(opt)
    return decoder



def get_model(opt):
    modality = opt['modality'].lower()
    input_size = []
    mapping = {
        'i': opt['dim_i'],
        'm': opt['dim_m'],
        'a': opt['dim_a']
    }
    for char in modality:
        assert char in mapping.keys()
        input_size.append(mapping[char])

    preEncoder, input_size = get_preEncoder(opt, input_size)
    encoder = get_encoder(opt, input_size, mapping, modality)

    if opt.get('intra_triplet', False) or opt['encoder_type'] == 'MME':
        joint_representation_learner = None
    else:
        joint_representation_learner = get_joint_representation_learner(opt)

    if len(opt['crit']) == 1:
        # only the main task: language generation
        if not opt.get('use_beam_decoder', False) and not opt.get('use_rl', False):
            assert opt['crit'][0] == 'lang'


    have_auxiliary_tasks = sum([(1 if item not in ['lang', 'tag'] else 0) for item in opt['crit']])
    auxiliary_task_predictor = Auxiliary_Task_Predictor(opt) if have_auxiliary_tasks else None

    decoder = get_decoder(opt)
    tgt_word_prj = nn.Linear(opt["dim_hidden"], opt["vocab_size"], bias=False)

    model = Seq2Seq(
        preEncoder = preEncoder,
        encoder = encoder,
        joint_representation_learner = joint_representation_learner,
        auxiliary_task_predictor = auxiliary_task_predictor,
        decoder = decoder,
        tgt_word_prj = tgt_word_prj,
        beam_decoder = beam_decoder,
        opt = opt
        )
    return model
