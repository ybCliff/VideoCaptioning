import argparse
import models.Constants as Constants
import os


def parse_opt():
    parser = argparse.ArgumentParser()
    # Modality Information
    parser.add_argument('-m', '--modality', type=str, default='ami',
                        help='the modality we want to use, any combination of \'a\' (audio), \'m\' (motion), and \'i\' (image)')
    parser.add_argument('--dim_a', type=int, default=644, help='dimension of the audio modality')
    parser.add_argument('--dim_m', type=int, default=2048, help='dimension of the motion modality')
    parser.add_argument('--dim_i', type=int, default=1536, help='dimension of the image modality')
    parser.add_argument('--dim_s', type=int, default=1)
    parser.add_argument('--feats_a_name', nargs='+', type=str,
                        default=['audio_vggish_60.hdf5', 'audio_boaw.hdf5', 'audio_fv.hdf5'])
    parser.add_argument('--feats_m_name', nargs='+', type=str, default=['motion_kinetics_16_8.hdf5'])
    parser.add_argument('--feats_i_name', nargs='+', type=str, default=['image_IRv2.hdf5'])
    parser.add_argument('--feats_s_name', nargs='+', type=str, default=[])

    # Path
    parser.add_argument('--base_dir', type=str, default='./data', help='base path to load corpus and features')
    parser.add_argument('--checkpoint_path', type=str, default='./data/experiments',
                        help='directory to store checkpointed models')
    parser.add_argument('--k_best_model', type=int, default=3, help='checkpoints with top-k performance will be saved')
    parser.add_argument('--save_checkpoint_every', type=int, default=1,
                        help='how often to save a model checkpoint (in epoch)?')
    parser.add_argument('-fewf', '--first_evaluate_whole_folder', default=False, action='store_true')

    # Dataset
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT', help='MSRVTT | Youtube2Text')
    #	- Corpus
    parser.add_argument('--info_corpus_name', type=str, default='msrvtt_info_corpus.pkl')
    parser.add_argument('--reference_name', type=str, default='msrvtt_refs.pkl')
    parser.add_argument('--max_len', type=int, default=30)

    parser.add_argument('--dim_hidden', type=int, default=512, help='size of the rnn hidden layer')
    parser.add_argument('-wc', '--with_category', default=False, action='store_true')
    parser.add_argument('--num_category', type=int, default=20)

    # Model Architecture
    parser.add_argument('--embed', default='m', type=str, help='embed specified modality into low-dimension space')
    parser.add_argument('--encoder_type', type=str, default='VOE', help='VOE | MLP | MSLSTM')
    parser.add_argument('--no_bn', default=False, action='store_true',
                        help='a BN layer will be placed between the encoder and decoder by default')
    parser.add_argument('--decoder_type', type=str, default='LSTM', help='LSTM | TopDown')
    # VOE Encoder
    parser.add_argument('-ngc', '--no_global_context', default=False, action='store_true')
    parser.add_argument('-nrc', '--no_regional_context', default=False, action='store_true')
    # MLP Encoder
    parser.add_argument('-MLP', '--MLP', default=False, action='store_true')
    # MSLSTM Encoder
    parser.add_argument('-MSLSTM', '--MSLSTM', default=False, action='store_true')

    parser.add_argument('--fusion_type', default='mean', type=str)

    # Model Hyper-parameters
    parser.add_argument('--encoder_dropout', type=float, default=0.5, help='strength of dropout in the encoder')
    parser.add_argument('--decoder_dropout', type=float, default=0.5, help='strength of dropout in the decoder')
    parser.add_argument('--dim_word', type=int, default=512, help='the encoding size of each token in the vocabulary')

    ######################### Training #########################
    parser.add_argument('-ss', '--scheduled_sampling', default=False, action='store_true')
    parser.add_argument('-sst', '--ss_type', default=1, type=int)
    parser.add_argument('-ssk', '--ss_k', default=100.0, type=float)
    parser.add_argument('-ssl', '--ss_linear', nargs='+', default=[0, 0.75], type=float)
    parser.add_argument('-ssp', '--ss_piecewise', nargs='+', default=[150, 0.95, 0.7], type=float)
    parser.add_argument('--ss_epochs', type=int, default=25)

    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('--tolerence', type=int, default=1000)
    parser.add_argument('--scope', type=str, default='')

    parser.add_argument('-no_cuda', '--no_cuda', default=False, action='store_true')
    parser.add_argument('-see', '--start_eval_epoch', type=int, default=0)

    ####################### Optimization #######################
    parser.add_argument('--grad_clip', type=float, default=5, help='clip gradients at this value')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-mlr', '--minimum_learning_rate', type=float, default=1e-5)
    parser.add_argument('--learning_rate_decay_every', type=int, default=1,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.994)
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay. strength of weight regularization')

    ##################################################################################

    parser.add_argument('--use_embedding', type=str, default='')
    parser.add_argument('--train_embedding', default=False, action='store_true')

    parser.add_argument('-bs', '--beam_size', type=int, default=1)
    parser.add_argument('-k', '--topk', type=int, default=1)
    parser.add_argument('-ba', '--beam_alpha', type=float, default=1.0)
    parser.add_argument('-bc', '--beam_candidate', type=int, default=5)
    parser.add_argument('-bsf', '--beam_search_file', type=str, default='beam_search_evaluate.csv')

    parser.add_argument('--att_mid_size', type=int, default=512)

    # sampling method
    parser.add_argument('--n_frames', type=int, default=8)
    parser.add_argument('--random_type', type=str, default='segment_random', help='segment_random | all_random')
    parser.add_argument('--equally_sampling', action='store_true')

    parser.add_argument('--metric_sum', nargs='+', type=int, default=[1, 1, 1, 1])

    parser.add_argument('--bn_momentum', default=0.1, type=float)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--start_eval_step', default=8000, type=int)
    parser.add_argument('--log', default=100, type=int)
    parser.add_argument('--save_checkpoint_every_step', default=100, type=int)

    parser.add_argument('--alr', default=5e-4, type=float)
    parser.add_argument('--decay', default=0.9, type=float)
    parser.add_argument('--amlr', default=5e-5, type=float)
    parser.add_argument('-wma', '--with_multimodal_attention', default=False, action='store_true')

    parser.add_argument('-bi', '--bidirectional', default=False, action='store_true')
    parser.add_argument('-op', '--object_path', type=str, default='')
    parser.add_argument('--dim_object', type=int, default=1000)
    parser.add_argument('--lamda', type=float, default=1.0)

    parser.add_argument('-all', '--all_caps_a_round', default=False, action='store_true')
    parser.add_argument('--decoder_hidden_init_type', type=int, default=0)
    parser.add_argument('--optim', type=str, default='adam', help='adam | rmsprop')
    parser.add_argument('--crit', nargs='+', type=str, default=['lang'], help='lang | obj | tag | length')
    parser.add_argument('--crit_name', nargs='+', type=str, default=['Cap Loss'])
    parser.add_argument('--crit_scale', nargs='+', type=float, default=[1.0])

    parser.add_argument('--use_tag', default=False, action='store_true')
    parser.add_argument('--dim_tag', type=int, default=32)
    parser.add_argument('--no_tag_emb', default=False, action='store_true')
    parser.add_argument('--last_tag', default=False, action='store_true')

    parser.add_argument('-se', '--shared_embedding', default=False, action='store_true')
    parser.add_argument('--load_feats_type', type=int, default=1,
                        help='load feats from the same (==0) frames_idx or different (not 0) frames_idx')

    # proposed method
    parser.add_argument('-kdwb', '--knowledge_distillation_with_bert', default=False, action='store_true')
    parser.add_argument('--bert_embeddings_name', type=str, default='word_embeddings.hdf5')
    parser.add_argument('--bert_embeddings', type=str,
                        default="/home/yangbang/VideoCaptioning/MSRVTT/feats/bert_embs_best_reduce_mean.hdf5")
    parser.add_argument('--dim_bert_embeddings', type=int, default=768)
    parser.add_argument('-ncpv', '--num_cap_per_vid', type=int, default=-1)

    parser.add_argument('--not_all', default=False, action='store_true')

    # multitask
    parser.add_argument('--triplet', default=False, action='store_true')
    parser.add_argument('--demand', nargs='+', default=['NOUN', 'VERB'])

    args = parser.parse_args()
    args.all_caps_a_round = True

    args.beta = [0, 1] if args.dataset == 'Youtube2Text' else [0.35, 0.9]

    if args.dataset == 'Youtube2Text':
        args.max_len = 20
        args.with_category = False

    if args.triplet:
        args.crit.append('triplet')
        args.crit_name.append('VT match loss')
        args.crit_scale.append(1.0)

    args.crit_key = [Constants.mapping[item.lower()] for item in args.crit]
    return args

'''
CUDA_VISIBLE_DEVICES=0 python train.py -ar -m mi -ss -wc 

CUDA_VISIBLE_DEVICES=0 python train.py -na -m mi -wc -method nv

AR - attribute generation 
CUDA_VISIBLE_DEVICES=3 python train.py -ar -method ag -m mi -wc --scope ag

python train.py -m mi -ar -wc -ubd -lpt "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_MI_seed920_ag/best/0047_176815_182718_186701_186558_185283.pth.tar" --method ag

python train.py -m mi -ar -wc -use_rl -lpt "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_MI_seed920_ag/best/0047_176815_182718_186701_186558_185283.pth.tar" --method ag
'''
