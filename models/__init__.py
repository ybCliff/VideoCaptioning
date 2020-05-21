
from .seq2seq import Seq2Seq
from .rnn import Hierarchical_Encoder#Encoder_Baseline, LSTM_Decoder
from .bert import BertEncoder, BertDecoder, NVADecoder, DirectDecoder, APDecoder, SignalDecoder, Signal3Decoder, Signal2Decoder, NVDecoder, MSDecoder, ARDecoder_with_attribute_generation, BeamDecoder
from .bert_pytorch import BertDecoder as BD
from .decoder import LSTM_Decoder, LSTM_GCC_Decoder, LSTM_Decoder_2stream, Top_Down_Decoder
from .encoder import Encoder_Baseline, Progressive_Encoder, SVD_Encoder, Input_Embedding_Layer, Semantics_Enhanced_IEL, HighWay_IEL, Encoder_HighWay, LEL
from .rnn import ENSEMBLE_Decoder 
from .joint_representation import Joint_Representaion_Learner
from .predictor import Auxiliary_Task_Predictor
import torch
import torch.nn as nn

def get_preEncoder(opt, input_size):
    preEncoder = None
    output_size = input_size.copy()

    if opt.get('use_preEncoder', False):
        output_size = [opt.get('dim_iel', opt['dim_hidden'])] * len(input_size)

        preEncoder = Input_Embedding_Layer(
                input_size=input_size,
                hidden_size=opt.get('dim_iel', opt['dim_hidden']), 
                name=opt['modality'].upper()
            )

    if opt.get('use_SEIEL', False):
        output_size = [opt['num_factor']] * len(input_size)
        preEncoder = Semantics_Enhanced_IEL(
                input_size=input_size,
                semantics_size=opt['dim_s'],
                nf=opt['num_factor'],
                name=opt['modality'],
                multiply=opt.get('SEIEL_multiply', False)
            )

    return preEncoder, output_size

def get_encoder(opt, input_size, mapping, modality):
    hidden_size = [opt['dim_hidden']] * len(modality)
    if opt['encoder_type'] == 'IPE':
        encoder = Hierarchical_Encoder(input_size = input_size, hidden_size = hidden_size, opt = opt)
    elif opt['encoder_type'] == 'IEL':
        encoder = HighWay_IEL(
                input_size=input_size, 
                hidden_size=hidden_size, 
                name=modality.upper(), 
                dropout=opt['encoder_dropout']
            )
    elif opt['encoder_type'] == 'LEL':
        encoder = LEL(
                input_size=input_size, 
                hidden_size=hidden_size, 
                name=modality.upper(), 
                dropout=opt['encoder_dropout']
            )
    elif opt['encoder_type'] == 'GRU':
        if opt.get('use_chain', False):
            encoder = Progressive_Encoder(
                    input_size=input_size,
                    output_size=hidden_size,
                    opt=opt,
                    return_gate_info=opt.get('return_gate_info', False)
                )
        else:
            auxiliary_pos = []
            for char in modality:
                auxiliary_for_this_input = opt.get('auxiliary_for_%s'%char, '')
                pos = []
                for c in auxiliary_for_this_input:
                    pos.append(modality.index(c))
                auxiliary_pos.append(pos)

            skip_info = opt.get('skip_info', [])
            if not len(skip_info): 
                skip_info = [0] * len(modality)
                opt['skip_info'] = skip_info

            from models.encoder import Encoder_Baseline_TwoStream
            if opt.get('two_stream', False):
                E = Encoder_Baseline_TwoStream
                if 'a' in modality:
                    hidden_size[modality.index('a')] = opt.get('dim_hidden_a', opt['dim_hidden'])
            else:
                E = Encoder_Baseline
                #E = Encoder_HighWay

            if opt.get('use_svd', False):
                encoder = SVD_Encoder(
                    input_size=input_size,
                    output_size=hidden_size,
                    name=modality.upper(),
                    auxiliary_pos=auxiliary_pos,
                    skip_info=skip_info,
                    return_gate_info=opt.get('return_gate_info', False),
                    num_factor=opt['num_factor']
                )
            else:    
                encoder = E(
                        input_size=input_size,
                        output_size=hidden_size,
                        name=modality.upper(),
                        auxiliary_pos=auxiliary_pos,
                        skip_info=skip_info,
                        return_gate_info=opt.get('return_gate_info', False),
                        opt=opt
                    )
    else:
        assert len(modality) == 1
        encoder = BertEncoder(feats_size = mapping[modality], config = opt)
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
        if opt.get('decoder_gcc', False):
            decoder = LSTM_GCC_Decoder(opt)
        elif opt.get('two_stream', False):
            decoder = LSTM_Decoder_2stream(opt)
        elif opt.get('top_down', False):
            decoder = Top_Down_Decoder(opt)
        else:
            decoder = LSTM_Decoder(opt)
    elif opt['decoder_type'] == 'ENSEMBLE':
        decoder = ENSEMBLE_Decoder(opt)
    elif opt['decoder_type'] == 'ARFormer':
        #decoder = BD(config=opt)
        if opt['method'] == 'ag':
            decoder = ARDecoder_with_attribute_generation(config=opt)
        else:
            decoder = BertDecoder(config=opt)
    else:
        if opt['method'] == 'mp':
            decoder = BertDecoder(config=opt)
        elif opt['method'] == 'nva':
            decoder = NVADecoder(config=opt)
        elif opt['method'] == 'direct':
            decoder = DirectDecoder(config=opt)
        elif opt['method'] == 'ap':
            decoder = APDecoder(config=opt)
        elif opt['method'] == 'signal':
            decoder = SignalDecoder(config=opt)
        elif opt['method'] == 'signal3':
            decoder = Signal3Decoder(config=opt)
        elif opt['method'] == 'signal2':
            decoder = Signal2Decoder(config=opt)
        elif opt['method'] == 'nv':
            decoder = NVDecoder(config=opt)
        elif opt['method'] == 'ms':
            decoder = MSDecoder(config=opt)
    return decoder

def get_beam_decoder(opt, embedding):
    if opt.get('use_beam_decoder', False):
        return BeamDecoder(opt, embedding)
    return None

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


    joint_representation_learner = get_joint_representation_learner(opt)

    if len(opt['crit']) == 1:
        # only the main task: language generation
        if not opt.get('use_beam_decoder', False):
            assert opt['crit'][0] == 'lang'


    have_auxiliary_tasks = sum([(1 if item not in ['lang', 'tag'] else 0) for item in opt['crit']])
    auxiliary_task_predictor = Auxiliary_Task_Predictor(opt) if have_auxiliary_tasks else None

    decoder = get_decoder(opt)
    tgt_word_prj = nn.Linear(opt["dim_hidden"], opt["vocab_size"], bias=False)
    beam_decoder = get_beam_decoder(opt, decoder.embedding)

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