import argparse
import models.Constants as Constants
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings

    parser.add_argument(
        '--cached_tokens',
        type=str,
        default='msr-all-idxs',
        help='Cached token file for calculating cider score \
                        during self critical training.')
    parser.add_argument(
        '--self_crit_after',
        type=int,
        default=-1,
        help='After what epoch do we start finetuning the CNN? \
                        (-1 = disable; never finetune, 0 = finetune from start)'
    )

    # Bert
    parser.add_argument('--dim_hidden', type=int, default=512, help='size of the rnn hidden layer')
    parser.add_argument('--num_hidden_layers_encoder', type=int, default=1)
    parser.add_argument('-ndl', '--num_hidden_layers_decoder', type=int, default=1)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--intermediate_size', type=int, default=2048)
    parser.add_argument('--hidden_act', type=str, default='gelu_new')
    parser.add_argument('--feat_act', type=str, default='gelu_new')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.5)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.0)
    parser.add_argument("--max_len", type=int, default=30, help='max length of captions(containing <sos>,<eos>)')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('-wc', '--with_category', default=False,action='store_true')
    parser.add_argument('--num_category', type=int, default=20)
    parser.add_argument('--watch', type=int, default=0)
    parser.add_argument('-pa', '--pos_attention', default=False, action='store_true')
    parser.add_argument('--enhance_input', type=int, default=0)
    parser.add_argument('--with_layernorm', default=False, action='store_true')
    parser.add_argument('--residual', nargs='+', type=int, default=[1, 1, 1, 1])
    parser.add_argument('--output_attentions', default=False, action='store_true')
    parser.add_argument('--output_hidden_states', default=False, action='store_true')

    # 
    parser.add_argument('-MSCA', '--multi_scale_context_attention', default=False, action='store_true')
    parser.add_argument('-qa', '--query_all', default=False, action='store_true')
    parser.add_argument('-wg', '--with_gate', default=False, action='store_true')
    

    # Model settings
    parser.add_argument('--use_preEncoder', default=False, action='store_true')
    parser.add_argument('-pet', '--preEncoder_type', default='linear', type=str)
    parser.add_argument('-pem', '--preEncoder_modality', default='m', type=str)

    parser.add_argument('--encoder_type', type=str, default='GRU', help='IPE | Former')
    parser.add_argument('--encoder_dropout', type=float, default=0.5, help='strength of dropout in the encoder')
    parser.add_argument('--no_encoder_bn', default=False, action='store_true')

    parser.add_argument('--decoder_type', type=str, default='LSTM', help='LSTM | ARFormer | NARFormer')
    parser.add_argument('--decoder_dropout', type=float, default=0.5)
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
    parser.add_argument('--year', type=int, default=2016)
    parser.add_argument('--tolerence', type=int, default=1000)
    parser.add_argument('--scope', type=str, default='')
    parser.add_argument('--loop_limit', type=int, default=1)

    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('-lpt', '--load_pretrained', type=str, default="")
    parser.add_argument('-fw', '--fix_weights', type=int, default=1, help='0: nothing; 1: encoder')
    ######################### Evaluation #########################
    parser.add_argument('-see', '--start_eval_epoch', type=int, default=0)


    ######################### Loading #########################
    parser.add_argument('--base_dir', type=str, default='/home/yangbang/VideoCaptioning/', help='base path to load corpus and features')
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT', help='MSRVTT | Youtube2Text')

    # features information
    parser.add_argument('-m', '--modality', type=str, default='ami')
    parser.add_argument('--dim_a', type=int, default=644)
    parser.add_argument('--dim_m', type=int, default=2048) #2048 3584
    parser.add_argument('--dim_i', type=int, default=2048) #1536
    parser.add_argument('--dim_s', type=int, default=1000)
    parser.add_argument('--dim_t', type=int, default=1300)

    parser.add_argument('--feats_a_name', nargs='+', type=str, default=['msrvtt_vggish_60.hdf5', 'vtt_boaw256.hdf5', 'fvdb_260.hdf5'])
    parser.add_argument('--feats_m_name', nargs='+', type=str, default=['kinetics_60.hdf5'])#, 'ECO.hdf5'])
    parser.add_argument('--feats_i_name', nargs='+', type=str, default=['R101.hdf5']) #'IRv2.hdf5'
    parser.add_argument('--feats_s_name', nargs='+', type=str, default=[]) #R101_logit1000.hdf5
    parser.add_argument('--feats_t_name', nargs='+', type=str, default=[])

    # Corpus information
    '''
    parser.add_argument('--info_json_name', type=str, default='info_pad_mask', help='path to the json file containing additional info and vocab')
    parser.add_argument('--caption_json_name', type=str, default='caption_pad_mask', help='path to the processed video caption json')
    parser.add_argument('--input_json_name', type=str, default='videodatainfo')
    parser.add_argument('--next_info_json_name', type=str, default='next_info_pad_mask')
    parser.add_argument('--all_caption_json_name', type=str, default='all_caption_pad_mask')
    '''
    parser.add_argument('--info_corpus_name', type=str, default='info_corpus')
    parser.add_argument('--reference_name', type=str, default='refs')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('-wct', '--word_count_threshold', type=int, default=2)

    ####################### Optimization #######################
    parser.add_argument('--grad_clip', type=float, default=5, help='clip gradients at this value')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-mlr', '--minimum_learning_rate', type=float, default=1e-5)
    parser.add_argument('--learning_rate_decay_every', type=int, default=1, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.994)
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay. strength of weight regularization')

    ######################### Saving #########################
    parser.add_argument('--checkpoint_path', type=str, default='/home/yangbang/VideoCaptioning/', help='directory to store checkpointed models')
    parser.add_argument('--checkpoint_path_name', default='0219save', type=str)
    parser.add_argument('--k_best_model', type=int, default=3, help='checkpoints with top-k performance will be saved')
    parser.add_argument('--save_checkpoint_every', type=int, default=1, help='how often to save a model checkpoint (in epoch)?')
    parser.add_argument('-fewf', '--first_evaluate_whole_folder', default=False, action='store_true')

    
##################################################################################
    
    # 是否使用预训练的词向量，是否fine-tune
    parser.add_argument('--use_embedding', type=str, default='')
    parser.add_argument('--train_embedding', default=False, action='store_true')


    # 测试时，是否使用beam search, 值>1说明使用
    parser.add_argument('-bs', '--beam_size', type=int, default=1)
    parser.add_argument('-k', '--topk', type=int, default=1)
    parser.add_argument('-ba', '--beam_alpha', type=float, default=1.0)
    parser.add_argument('-bc', '--beam_candidate', type=int, default=5)
    parser.add_argument('-bsf', '--beam_search_file', type=str, default='beam_search_evaluate.csv')

    # NAR
    parser.add_argument('--length_beam_size', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--use_gold_target_len', default=False)
    parser.add_argument('-ut', '--use_teacher', default=False, action='store_true')
    parser.add_argument('-teacher_path', type=str, default="/home/yangbang/VideoCaptioning/1007save/MSRVTT/IPE_ARFormer/AMIs_Seed0_EBN_WC20_SS1_100_70_FromScratchNoWatch_pjDrop/")
    parser.add_argument('-teacher_name', type=str, default='0_0455_186967_192003_182832_192057_191210.pth.tar')

    parser.add_argument('--att_mid_size', type=int, default=512)

    # sampling method
    parser.add_argument('--n_frames', type=int, default=8)
    parser.add_argument('--random_type', type=str, default='segment_random', help='segment_random | all_random')
    parser.add_argument('--equally_sampling', action='store_true')
    
    parser.add_argument('--metric_sum', nargs='+', type=int, default=[1, 1, 1, 1])


    parser.add_argument('--bn_momentum', default=0.1, type=float)

    parser.add_argument('--transfer_validation', default=False, action='store_true')
    parser.add_argument('--transfer_dataset', default='Youtube2Text', type=str)
    parser.add_argument('--fusion_type', default='addition', type=str)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--start_eval_step', default=8000, type=int)
    parser.add_argument('--log', default=100, type=int)
    parser.add_argument('--save_checkpoint_every_step', default=100, type=int)

    parser.add_argument('--AGRU_input_type', default='concat', type=str)

    parser.add_argument('--alr', default=5e-4, type=float)
    parser.add_argument('--decay', default=0.9, type=float)
    parser.add_argument('--amlr', default=5e-5, type=float)
    parser.add_argument('-wma', '--with_multimodal_attention', default=False, action='store_true')

    
    parser.add_argument('--skip_info', nargs='+', type=int, default=[])
    parser.add_argument('-afa', '--auxiliary_for_a', type=str, default='')
    parser.add_argument('-afm', '--auxiliary_for_m', type=str, default='')
    parser.add_argument('-afi', '--auxiliary_for_i', type=str, default='')

    parser.add_argument('--addition', default=False, action='store_true')
    parser.add_argument('--temporal_concat', default=False, action='store_true')
    
    parser.add_argument('-dg', '--decoder_gcc', default=False, action='store_true')
    parser.add_argument('-ts', '--two_stream', default=False, action='store_true')
    parser.add_argument('-td', '--top_down', default=False, action='store_true')

    parser.add_argument('-gs', '--gated_sum', default=False, action='store_true')

    parser.add_argument('--use_svd', default=False, action='store_true')
    parser.add_argument('-nf', '--num_factor', type=int, default=512)
    
    parser.add_argument('--dim_hidden_a', type=int, default=512)
    parser.add_argument('-bi', '--bidirectional', default=False, action='store_true')
    parser.add_argument('-ga', '--gate_attention', default=False, action='store_true')
    

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

    parser.add_argument('--use_chain', default=False, action='store_true')
    parser.add_argument('--chain_both', default=False, action='store_true')

    parser.add_argument('--gcc_addition', default=False, action='store_true')
    parser.add_argument('--eh_init', default=False, action='store_true')

    
    parser.add_argument('-tse', '--task_specific_embedding', default=False, action='store_true')
    
    parser.add_argument('--use_SEIEL', default=False, action='store_true')
    parser.add_argument('--SEIEL_multiply', default=False, action='store_true')

    parser.add_argument('--others', default=False, action='store_true')
    parser.add_argument('--corpus_name', type=str, default='msvd_corpus_glove')
    parser.add_argument('--varlstm', default=False, action='store_true')
    parser.add_argument('-vid', '--varlstm_id', type=float, default=0.3)
    parser.add_argument('-vhd', '--varlstm_hd', type=float, default=0.0)

    parser.add_argument('--mylstm', default=False, action='store_true')

    parser.add_argument('-ar', '--ar', default=False, action='store_true', help='autoregressive')
    parser.add_argument('-na', '--na', default=False, action='store_true', help='non-autoregressive')
    parser.add_argument('-method', '--method', type=str, default='mp', help='mp: mask-predict | nva: proposed method')
    parser.add_argument('-se', '--shared_embedding', default=False, action='store_true')
    parser.add_argument('--load_feats_type', type=int, default=0, help='load feats from the same (==0) frames_idx or different (not 0) frames_idx')

    # proposed method
    parser.add_argument('--demand', nargs='+', type=str, default=['VERB', 'NOUN'], help='pos_tag we want to focus on')
    parser.add_argument('-nstt', '--nav_source_target_type', type=str, default='noise', help='noise | gt | mp')
    parser.add_argument('--reverse_prob', type=float, default=0.2)
    parser.add_argument('--select_out_ratio', type=float, default=0.0)
    
    parser.add_argument('--use_eos', default=False, action='store_true')
    parser.add_argument('--use_kl', default=False, action='store_true')
    parser.add_argument('--no_signal', default=False, action='store_true')
    parser.add_argument('--bow_loss', default=False, action='store_true')
    

    parser.add_argument('--dist', type=int, default=0)
    parser.add_argument('-nvw', '--nv_weights', nargs='+', type=float, default=[0.8, 1.0])
    parser.add_argument('-ms', '--multiscale', type=int, default=3)

    parser.add_argument('--irv2c3dk', default=False, action='store_true')
    parser.add_argument('--irv2c3ds', default=False, action='store_true')
    parser.add_argument('--irv2c3dsl', default=False, action='store_true')
    parser.add_argument('--irv2c3dkl', default=False, action='store_true')
    parser.add_argument('--google', default=False, action='store_true')
    parser.add_argument('--r152', default=False,action='store_true')
    parser.add_argument('--eco', default=False, action='store_true')

    parser.add_argument('-kdwb', '--knowledge_distillation_with_bert', default=False, action='store_true')
    parser.add_argument('--bert_embeddings_name', type=str, default='word_embeddings.hdf5')
    parser.add_argument('--dim_bert_embeddings', type=int, default=768)

    parser.add_argument('--paradigm', type=str, default='mp')
    parser.add_argument('-ma', '--multitask_attribute', default=False, action='store_true')
    parser.add_argument('-ncpv', '--num_cap_per_vid', type=int, default=-1)

    # testing beam candidate selection

    parser.add_argument('-ubd', '--use_beam_decoder', default=False, action='store_true')
    parser.add_argument('--bd_parameters', nargs='+', type=float, default=[6, 6, 1.0, 3], \
        help='[beam_size, topk, beam_alpha, num_positive], prepare data to train beam decoder')

    # for zhangcan
    parser.add_argument('-upl', '--use_pan_lite', default=False, action='store_true')

    # reinforement learning
    parser.add_argument('-use_rl', '--use_rl', default=False, action='store_true')    
    parser.add_argument('--rl_cached_file', type=str, default='rl_cached')    
    

    args = parser.parse_args()
    args.all_caps_a_round = True

    args.beta = [0, 1] if args.dataset == 'Youtube2Text' else [0.35, 0.9]

    if args.dataset == 'Youtube2Text':
        args.max_len = 20
        args.word_count_threshold = 0
        args.feats_a_name = [''] if 'a' not in args.modality else ['msvd_vggish.hdf5', 'msvd_boaw256.hdf5', 'msvd_fvdb_260.hdf5']
        args.with_category = False

    if args.ar:
        args.encoder_type = 'IEL'#'IEL'LEL
        args.decoder_type = 'ARFormer'
        args.temporal_concat = True
        

    if args.na:
        args.crit = ['lang', 'length']
        args.crit_name = ['Cap Loss', 'Length Loss']
        if args.method == 'ms':
            args.crit_scale = [1.0 / args.multiscale, 1.0]
            args.enhance_input = 0
        else:
            args.crit_scale = [1.0, 1.0]
            args.enhance_input = 2
        args.encoder_type = 'IEL' #'IEL', LEL
        args.decoder_type = 'NARFormer'
        args.temporal_concat = True
        #args.addition = True
        args.masking_ratio = 0.5

        args.no_duplicate = True
        args.use_kl = True
        args.no_signal = True

        if args.method == 'signal3' or args.method == 'nv' or args.method == 'ms':
            args.visual_tag = Constants.BOS
            args.nonvisual_tag = Constants.EOS
            args.revision_tag = Constants.MASK

            args.nv_scale = 100

        
    if args.ar or args.na:
        args.load_feats_type = 1
        if args.dataset.lower() == 'vatex':
            args.modality = 'm'
            args.feats_m_name = ['I3D.hdf5']
            args.dim_m = 1024
            args.feats_i_name = []
            args.feats_a_name = []
            args.max_len = 30
            args.word_count_threshold = 2
            args.with_category = False

            args.dim_hidden = 1024
            args.num_attention_heads = 16
            args.intermediate_size = 4096
            args.epochs = 100
        else:
            if args.irv2c3dk:
                args.feats_m_name = ['c3d_fc6_16_8_kinetics.hdf5']#['c3d_60_pool5.hdf5']#['c3d_60_fc6.hdf5']
                args.feats_i_name = ['IRv2.hdf5']
                args.dim_m = 4096 #512# + 1536
                args.dim_i = 1536
                args.modality = 'mi'
                args.load_feats_type = 1
                args.scope = 'irv2c3dk'
            elif args.irv2c3ds:
                args.feats_m_name = ['c3d_fc6_16_8_sports1m.hdf5']#['c3d_60_pool5.hdf5']#['c3d_60_fc6.hdf5']
                args.feats_i_name = ['IRv2.hdf5']
                args.dim_m = 4096 #512# + 1536
                args.dim_i = 1536
                args.modality = 'mi'
                args.load_feats_type = 1
                args.scope = 'irv2c3ds'
            elif args.irv2c3dsl:
                args.feats_m_name = ['c3d_logits_16_8_sports1m.hdf5']#['c3d_60_pool5.hdf5']#['c3d_60_fc6.hdf5']
                args.feats_i_name = ['IRv2.hdf5']
                args.dim_m = 487 #512# + 1536
                args.dim_i = 1536
                args.modality = 'mi'
                args.load_feats_type = 1
                args.scope = 'irv2c3dsl'
            elif args.irv2c3dkl:
                args.feats_m_name = ['c3d_logits_16_8_kinetics.hdf5']#['c3d_60_pool5.hdf5']#['c3d_60_fc6.hdf5']
                args.feats_i_name = ['IRv2.hdf5']
                args.dim_m = 400 #512# + 1536
                args.dim_i = 1536
                args.modality = 'mi'
                args.load_feats_type = 1
                args.scope = 'irv2c3dkl'
            elif args.google:
                args.feats_i_name = ['G.hdf5']
                args.dim_i = 1024
                args.modality = 'i'
                args.scope = 'google'
            elif args.r152:
                args.feats_i_name = ['R152.hdf5']
                args.dim_i = 2048
                args.modality = 'i'
                args.scope = 'r152'
            elif args.eco:
                args.feats_m_name = ['ECO.hdf5']
                args.feats_a_name = []
                args.dim_m = 1536
                args.modality = 'mi'
                args.feats_i_name = ['resnet101_60.hdf5']
                args.scope = 'eco2'
            else:
                args.feats_m_name = ['kinetics_16_8.hdf5']
                args.feats_i_name = ['resnet101_60.hdf5']
                args.feats_a_name = ['msrvtt_vggish_60.hdf5', 'vtt_boaw256.hdf5', 'fvdb_260.hdf5']

        if args.use_tag:
            #args.feats_t_name = ['attribute_tag.hdf5']
            '''
            args.crit.append('attr')
            args.crit_name.append('Attr Loss')
            args.crit_scale.append(1)
            args.scope = args.scope + 'tag'
            args.dim_t = 1000
            '''
            args.feats_t_name = args.feats_a_name
            args.scope = args.scope + 'AudioEmb'
            args.dim_t = 644

        if args.multitask_attribute:
            args.crit.append('attr2')
            args.crit_name.append('Attr2 Loss')
            args.crit_scale.append(1)
            args.scope = args.scope + 'multitask'
            args.dim_t = 1000

        if args.knowledge_distillation_with_bert:
            args.crit.append('dist')
            args.crit_name.append('Dist Loss')
            args.crit_scale.append(1)
            args.scope = args.scope + '_bertKD'

        if args.num_cap_per_vid != -1:
            args.scope += 'numCap%d'%args.num_cap_per_vid

        if args.use_beam_decoder:
            args.crit = ['beam']
            args.crit_name = ['Beam Loss']
            args.crit_scale = [1]
            args.scope += 'ubd'

        
        #pass
    # for zhangcan
    if args.use_pan_lite:
        assert args.dataset == 'Youtube2Text'
        args.modality = 'i'
        args.dim_i = 2048
        args.feats_i_name = ['pan_lite.hdf5']
        args.feats_m_name = args.feats_a_name = []
        args.dim_m = args.dim_a = 1
        args.load_feats_type = 0
        args.scope += 'pan_lite'

    # reinforcement learning
    args.rl_cached_file = os.path.join(args.base_dir, args.dataset, args.rl_cached_file + '_%d'%args.word_count_threshold)
    if args.use_rl:
        args.crit = ['self_crit']
        args.crit_name = ['Reward']
        args.crit_scale = [1.0]
        args.scope = args.scope + '_rl'
        args.all_caps_a_round = False
        args.tolerence = 10
        args.k_best_model = 1
        args.standard = ['CIDEr']

    args.crit_key = [Constants.mapping[item.lower()] for item in args.crit]
    return args


'''
CUDA_VISIBLE_DEVICES=0 python train.py -ar -m mi -ss -wc 
-m 表示使用什么模态信息 -ss表示使用 scheduled sampling, -wc是使用类别信息

非自回归训练：
CUDA_VISIBLE_DEVICES=0 python train.py -na -m mi -wc -method nv

AR - attribute generation 
CUDA_VISIBLE_DEVICES=3 python train.py -ar -method ag -m mi -wc --scope ag

python train.py -m mi -ar -wc -ubd -lpt "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_MI_seed920_ag/best/0047_176815_182718_186701_186558_185283.pth.tar" --method ag

python train.py -m mi -ar -wc -use_rl -lpt "/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_ARFormer/EBN1_SS0_NDL1_WC20_MI_seed920_ag/best/0047_176815_182718_186701_186558_185283.pth.tar" --method ag
'''