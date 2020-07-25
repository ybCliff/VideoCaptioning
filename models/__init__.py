import torch
import torch.nn as nn
from .encoder import Input_Embedding_Layer, Visual_Oriented_Encoder, MLP, Encoder_Baseline
from .fusion import Joint_Representaion_Learner
from .decoder import LSTM_Decoder, Top_Down_Decoder
from .seq2seq import Seq2Seq


def get_preEncoder(opt):
    '''
        preEncoder is to embed features into low-dimention space
        e.g., 4096-D C3D features can be mapped into 512-D embeddings 
        by setting use_preEncoder=True and preEncoder_modality='m'
    '''
    modality = opt['modality'].lower()
    input_size = []
    mapping = {
        'i': opt['dim_i'],  # image
        'm': opt['dim_m'],  # motion
        'a': opt['dim_a'],  # audio
    }
    for char in modality:
        assert char in mapping.keys()
        input_size.append(mapping[char])

    preEncoder = None
    output_size = input_size.copy()

    if opt.get('use_preEncoder', False):
        pem = opt.get('preEncoder_modality', '')
        if pem:
            # only the specific modalities will be processed
            skip_info = [1] * len(input_size)
            for char in pem:
                pos = modality.index(char)
                skip_info[pos] = 0
                output_size[pos] = opt['dim_hidden']
        else:
            # all features will be processed
            skip_info = [0] * len(input_size)
            output_size = [opt['dim_hidden']] * len(input_size)
            
        preEncoder = Input_Embedding_Layer(
                input_size=input_size,
                hidden_size=opt['dim_hidden'], 
                skip_info=skip_info,
                name=modality.upper()
            )

    return preEncoder, output_size


def get_encoder(opt, input_size):
    modality = opt['modality'].lower()
    hidden_size = [opt['dim_hidden']] * len(input_size)
    if opt['encoder_type'] == 'VOE':
        encoder = Visual_Oriented_Encoder(input_size=input_size, hidden_size=hidden_size, opt = opt)
    elif opt['encoder_type'] == 'MLP':
        encoder = MLP(sum(input_size), opt['dim_hidden'], 'a' in modality)
    elif opt['encoder_type'] == 'MSLSTM':
        encoder = Encoder_Baseline(input_size=input_size, output_size=hidden_size, name=modality.upper(), encoder_type='mslstm')
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
    return Joint_Representaion_Learner(feats_size, fusion_type=opt['fusion_type'], with_bn=not opt['no_bn'])


def get_decoder(opt):
    if opt['decoder_type'] == 'LSTM':
        decoder = LSTM_Decoder(opt)
    elif opt['decoder_type'] == 'TopDown':
        decoder = Top_Down_Decoder(opt)
    return decoder


def get_model(opt):
    preEncoder, input_size = get_preEncoder(opt)
    encoder = get_encoder(opt, input_size)
    joint_representation_learner = get_joint_representation_learner(opt)

    if len(opt['crit']) == 1:
        # only the main task: language generation
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
        opt = opt
        )
    return model
