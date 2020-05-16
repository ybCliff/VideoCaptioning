import os
import time
from tqdm import tqdm
import json
import argparse


parser = argparse.ArgumentParser()
# Data input settings
parser.add_argument('--loop_limit', type=int, default=1)
parser.add_argument('--stop_count', type=int, default=1000)
parser.add_argument('--save_checkpoint_every', type=int, default=1)
parser.add_argument('--start_eval_epoch', type=int, default=130)
parser.add_argument('--scope', type=str, default='')
#parser.add_argument('--modality', type=str, default='ic')
#parser.add_argument('--through_encoder', nargs='+', type=int, default=[1, 1, 1])
#parser.add_argument('--mean_feats', nargs='+', type=int, default=[0, 0, 0])
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--option', type=int ,default=0)
parser.add_argument('--d', type=str, default='MSRVTT')
parser.add_argument('--m', type=str, default='ica')
parser.add_argument('--S2ADRM_type', type=str, default='zi')
parser.add_argument('--load_pretrained', type=str, default='\'\'')
parser.add_argument('--c3d_feats_name', type=str, default='c3d_60_fc6.hdf5')
parser.add_argument('--first_evaluate_whole_folder', action='store_true')
parser.add_argument('--usePAA', action='store_true')
parser.add_argument('--acoustic', nargs='+', type=int, default=[256, 260])

parser.add_argument('-a', '--activation', default='tanh', type=str)
parser.add_argument('-at', '--activation_type', default='acc', type=str)
parser.add_argument('--type_PAA', default=0, type=int)
parser.add_argument('--with_modality_att', default=False, action='store_true')
parser.add_argument('--merge_type', default='o', type=str)
parser.add_argument('--random_type', default='segment_random', type=str)
parser.add_argument('--equally_sampling', default=False, action='store_true')
parser.add_argument('--att_mid_size', default=256, type=int)
parser.add_argument('--use_bn', default=False, action='store_true')
parser.add_argument('--fusion_type', default='addition', type=str)

parser.add_argument('--encoder_type', default='gru', type=str)
parser.add_argument('--decoder_type', default='lstm', type=str)
parser.add_argument('--use_preEncoder', default=False, action='store_true')
parser.add_argument('--preEncoder_type', default='linear', type=str)
parser.add_argument('--preEncoder_modality', default='\'\'', type=str)
parser.add_argument('--concat_before_att', default=False, action='store_true')
parser.add_argument('--dim_encoder_hiddenC', default=512, type=int)

parser.add_argument('--scheduled_sampling', default=False, action='store_true')
parser.add_argument('--ss_type', default=1, type=int)
parser.add_argument('--ss_k', default=100.0, type=float)
parser.add_argument('--ss_linear', nargs='+', default=[100, 0.7], type=float)
parser.add_argument('--ss_piecewise', nargs='+', default=[150, 0.95, 0.65], type=float)

parser.add_argument('--seed', default=1002, type=int)
parser.add_argument('--mo', default='\'\'', type=str)
parser.add_argument('--mi', default='\'\'', type=str)
parser.add_argument('-alm', '--all_level_modality', nargs='+', type=int, default=[0, 0, 0, 0])
parser.add_argument('--n_frames', default=8, type=int)
parser.add_argument('--dim_guidance', default=128, type=int)
parser.add_argument('--guidance_type', default='full', type=str)
parser.add_argument('--no_DMHE_bn', default=False, action='store_true')
parser.add_argument('--att_dropout', default=0.0, type=float)
parser.add_argument('--together', default=False, action='store_true')

parser.add_argument('--forget_bias', default=0.6, type=float)
parser.add_argument('--keyword', default='\'\'', type=str)
parser.add_argument('--no_save_best', default=False, action='store_true')
parser.add_argument('--cheat', default=False, action='store_true')
parser.add_argument('--bidirection', default=False, action='store_true')
parser.add_argument('--dim_encoder_hidden', default=512, type=int)
parser.add_argument('--grad_clip', default=5, type=float)
parser.add_argument('--twostream', default=False, action='store_true')
parser.add_argument('--use_MA', default=False,action='store_true')
parser.add_argument('--connect_type', default='Direct', type=str)
parser.add_argument('--global_type', default='Flow', type=str)
parser.add_argument('--ss_wise', default=False, action='store_true')
parser.add_argument('--baseline_addition', default=False, action='store_true')
parser.add_argument('--dim_global', default=128, type=int)
parser.add_argument('--category_type', default=1, type=int)
args = parser.parse_args()
#loop_limit = args.loop_limit
if '.hdf5' in args.c3d_feats_name:
    if args.d == 'MSRVTT':
        args.c3d_feats_name = 'msrvtt_' + args.c3d_feats_name
    else:
        args.c3d_feats_name = 'msvd_' + args.c3d_feats_name
#dataset = 'Youtube2Text'
checkpoint_path_name = '1006save'
model = 'DFM_Model'
k_best_model = 3
save_model_limit = 50
teacher_prob = 1
learning_rate_decay_rate = 0.994
learning_rate_decay_every = 1
beam_size = 1
train_type = '1'
batch_size = {'MSRVTT': 128, 'Youtube2Text': 64}
word_count_threshold = {'MSRVTT': 2, 'Youtube2Text': 0}
max_len = {'MSRVTT': 30, 'Youtube2Text': 20}
train_max_len = {'MSRVTT': 30, 'Youtube2Text': 20}
dropout_p = {'MSRVTT': 0.5, 'Youtube2Text': 0.5}
epochs = {'MSRVTT': 500, 'Youtube2Text': 500}
learning_rate = {'MSRVTT': 1e-3, 'Youtube2Text': 1e-3}
discriminative_feats_name = {'MSRVTT': '\'\'', 'Youtube2Text': '\'\''}
dim_d = {'MSRVTT': 1024, 'Youtube2Text': 1024}
acoustic_feats_name = {'MSRVTT': ['msrvtt_vggish.hdf5'], 'Youtube2Text': ['msvd_vggish.hdf5']}#msvd_mfcc_samples.hdf5
feats_name = {'MSRVTT': ['msrvtt_IRv2.hdf5'], 'Youtube2Text': ['msvd_IRv2.hdf5']}
dim_acoustic = {'MSRVTT': 128, 'Youtube2Text': 128}
input_json_name = {'MSRVTT': 'videodatainfo', 'Youtube2Text': 'videodatainfo'}
info_json_name = {'MSRVTT': 'info_pad_mask', 'Youtube2Text': 'info'}
caption_json_name = {'MSRVTT': 'caption_pad_mask', 'Youtube2Text': 'caption'}


acoustic_post_name = ''
if 'a' in args.m:
    acoustic_post_name = '_Av'
    if args.acoustic[0]:
        acoustic_feats_name['MSRVTT'].append('vtt_boaw%d.hdf5' % args.acoustic[0])
        acoustic_feats_name['Youtube2Text'].append('msvd_boaw%d.hdf5' % args.acoustic[0])
        dim_acoustic['MSRVTT'] += args.acoustic[0]
        dim_acoustic['Youtube2Text'] += args.acoustic[0]
        acoustic_post_name += 'b%d' % args.acoustic[0]

    if args.acoustic[1]:
        acoustic_feats_name['MSRVTT'].append('fvdb_%d.hdf5' % args.acoustic[1])
        acoustic_feats_name['Youtube2Text'].append('msvd_fvdb_%d.hdf5' % args.acoustic[1])
        dim_acoustic['MSRVTT'] += args.acoustic[1]
        dim_acoustic['Youtube2Text'] += args.acoustic[1]
        acoustic_post_name += 'f%d' % args.acoustic[1]

discriminative_feats_dir = {'MSRVTT': '\'\'', 'Youtube2Text': '\'\''}
num_topics = 20
ltm_dir = '\'\''

if 'c3d' in args.c3d_feats_name:
    #dim_c3d=2048
    dim_c3d = 4096 if 'fc6' in args.c3d_feats_name else 512
else:
    dim_c3d = 2048

if args.option == 0:
    #dataset = 'MSRVTT'
    dataset = args.d
    modality_list = [args.m] * 1
    modality_zo_list = [args.mo] * 1
    modality_zi_list = [args.mi] * 1
    through_encoder_list = [[1, 1, 1, 1]] * 12
    mean_feats_list = [[0, 0, 0, 0]] * 12
    att_info_list = [[1, 0, 0, 0, 0]] * 12 
    
    if args.equally_sampling: rt = 'ES'
    else: rt = 'AR' if args.random_type == 'all_random' else 'SR'

    #prename = '%s_%s%s%s%d_wMA%d' % (rt, args.activation.upper(), args.activation_type, args.fusion_type.upper(), args.att_mid_size, 1 if args.with_modality_att else 0)
    prename = '%s%dwt%d_G%s_R%s%s%s%s%s' % (rt,args.n_frames, word_count_threshold[dataset], args.global_type, args.connect_type, ('_pE%s'%args.preEncoder_modality) if args.use_preEncoder else '', '_wise' if args.ss_wise else '', ('_SS%d'%args.ss_type) if args.scheduled_sampling else '', '_together' if args.together else '')
    if args.ss_type == 1 and args.scheduled_sampling:
    	prename += '_%d_%d' % (args.ss_linear[0], int(100 * args.ss_linear[1]))
    scope_list = [ 
        args.scope + acoustic_post_name, 
        ]
    if args.d == 'Youtube2Text':
    	tmp = ' --useS2ADRM --concat_before_att '
    else:
    	tmp = ' --useS2ADRM --concat_before_att --use_ltm '
    scope_list = [prename+item for item in scope_list]
    info_list = [
            #' --useS2ADRM --use_preEncoder --preEncoder_type linear --dim_encoder_hiddenI 512 --dim_encoder_hiddenC 512 ',
            #' --useS2ADRM --scheduled_sampling --ss_k 70 ',
            tmp
            ]
    
    assert len(modality_list) == len(scope_list)
    assert len(scope_list) == len(info_list)


#forget_bias = [(item / 10) for item in range(2, 21, 2)]
#for fb in forget_bias:
#    args.forget_bias = fb

#dim_encoder_hiddenC = [item for item in range(128, 1025, 128)]
#for dc in dim_encoder_hiddenC:
#    args.dim_encoder_hiddenC = dc

#nframes = [item for item in range(5, 21)]
#for nf in nframes:
#    args.n_frames = nf
print('loop_limit: ', args.loop_limit)
for i in range(args.loop_limit):
    for j in range(len(info_list)):
        scope = scope_list[j]
        modality = modality_list[j]
        modality_zo = modality_zo_list[j]
        modality_zi = modality_zi_list[j]
        print(modality, modality_zo, modality_zi)
        if not modality: modality = '\'\''
        if not modality_zo: modality_zo = '\'\''
        if not modality_zi: modality_zi = '\'\''
        through_encoder = through_encoder_list[j]
        mean_feats = mean_feats_list[j]
        info = info_list[j]
        att_info = att_info_list[j]
        #######

        tmp = ' --first_evaluate_whole_folder ' if args.first_evaluate_whole_folder else ''
        paa = ' --usePAA ' if args.usePAA else ''
        if args.with_modality_att:
            tmp += ' --with_modality_att '
        if args.equally_sampling:
            tmp += ' --equally_sampling '
        if args.use_bn:
            tmp += ' --use_bn '
        if args.use_preEncoder:
            tmp += ' --use_preEncoder '
        if args.concat_before_att:
            tmp += ' --concat_before_att '
        if args.scheduled_sampling:
            tmp += ' --scheduled_sampling '
        if not args.no_DMHE_bn:
            tmp += ' --no_DMHE_bn '
        if args.together:
            tmp += ' --together '
        if args.no_save_best:
            tmp += ' --no_save_best '
        if args.cheat:
            tmp += ' --cheat '
        if args.bidirection:
            tmp += ' --bidirection '
        if args.twostream:
            tmp += ' --2stream '
        if args.use_MA:
        	tmp += ' --use_MA '
        if args.ss_wise:
            tmp += ' --ss_wise '
        if args.baseline_addition:
        	tmp += ' --baseline_addition '
        op = 'CUDA_VISIBLE_DEVICES='+ args.gpu\
            +' python train.py '\
            +' --scope ' + scope\
            +' --dataset ' + dataset\
            +' --batch_size ' + str(batch_size[dataset])\
            +' --max_len ' + str(max_len[dataset])\
            +' --word_count_threshold ' + str(word_count_threshold[dataset])\
            +' --k_best_model ' + str(k_best_model)\
            +' --save_model_limit ' + str(save_model_limit)\
            +' --teacher_prob ' + str(teacher_prob)\
            +' --learning_rate ' + str(learning_rate[dataset])\
            +' --model DFM_Model '\
            +' --modality ' + modality \
            +' --through_encoder ' + ' '.join([str(k) for k in through_encoder]) \
            +' --mean_feats ' + ' '.join([str(k) for k in mean_feats])\
            +' --att_info ' + ' '.join([str(k) for k in att_info])\
            +' --input_dropout_p ' + str(dropout_p[dataset])\
            +' --output_dropout_p ' + str(dropout_p[dataset])\
            +' --epochs ' + str(epochs[dataset])\
            +' --dim_d ' + str(dim_d[dataset])\
            +' --discriminative_feats_name ' + discriminative_feats_name[dataset]\
            +' --acoustic_feats_name ' + ' '.join(acoustic_feats_name[dataset])\
            +' --dim_acoustic ' + str(dim_acoustic[dataset])\
            +' --loop_limit ' + str(i+1)\
            +' --loop_start ' + str(i)\
            +' --save_checkpoint_every ' + str(args.save_checkpoint_every)\
            +' --stop_count ' + str(args.stop_count)\
            +' --start_eval_epoch ' + str(args.start_eval_epoch)\
            +' --train_type ' + str(train_type)\
            +' --learning_rate_decay_rate ' + str(learning_rate_decay_rate)\
            +' --learning_rate_decay_every ' + str(learning_rate_decay_every)\
            +' --beam_size ' + str(beam_size)\
            +' --discriminative_feats_dir ' + discriminative_feats_dir[dataset]\
            +' --train_max_len ' + str(train_max_len[dataset])\
            +' --checkpoint_path_name ' + checkpoint_path_name\
            +' --model ' + model\
            +' --ltm_dir '+ ltm_dir\
            +' --num_topics '+ str(num_topics)\
            +' --S2ADRM_type '+args.S2ADRM_type\
            +' --load_pretrained ' + args.load_pretrained\
            +' --c3d_feats_name ' + args.c3d_feats_name\
            +' --dim_c3d ' + str(dim_c3d)\
            +' --activation ' + args.activation\
            +' --activation_type ' + args.activation_type\
            +' --type_PAA ' + str(args.type_PAA)\
            +' --merge_type ' + args.merge_type\
            +' --random_type ' + args.random_type\
            +' --att_mid_size ' + str(args.att_mid_size)\
            +' --fusion_type ' + args.fusion_type\
            +' --encoder_type ' + args.encoder_type\
            +' --decoder_type ' + args.decoder_type\
            +' --seed ' + str(args.seed)\
            +' --modality_zo ' + modality_zo\
            +' --modality_zi ' + modality_zi\
            +' --all_level_modality ' + ' '.join([str(k) for k in args.all_level_modality])\
            +' --n_frames ' + str(args.n_frames)\
            +' --dim_guidance ' + str(args.dim_guidance)\
            +' --guidance_type ' + args.guidance_type\
            +' --feats_name ' + ' '.join(feats_name[dataset])\
            +' --preEncoder_type ' + args.preEncoder_type\
            +' --preEncoder_modality ' + args.preEncoder_modality\
            +' --dim_encoder_hiddenC ' + str(args.dim_encoder_hiddenC)\
            +' --ss_k ' + str(args.ss_k)\
            +' --ss_type ' + str(args.ss_type)\
            +' --ss_linear ' + ' '.join([str(k) for k in args.ss_linear])\
            +' --ss_piecewise ' + ' '.join([str(k) for k in args.ss_piecewise])\
            +' --input_json_name ' + input_json_name[dataset]\
            +' --info_json_name ' + info_json_name[dataset]\
            +' --caption_json_name ' + caption_json_name[dataset]\
            +' --att_dropout ' + str(args.att_dropout)\
            +' --forget_bias ' + str(args.forget_bias)\
            +' --keyword ' + args.keyword\
            +' --dim_encoder_hidden ' + str(args.dim_encoder_hidden)\
            +' --grad_clip ' + str(args.grad_clip)\
            +' --connect_type ' + args.connect_type\
            +' --global_type ' + args.global_type\
            +' --dim_global ' + str(args.dim_global)\
            +' --category_type ' + str(args.category_type)\
            + paa\
            + tmp\
            + info

        print(op)
        
        os.system(op)


'''
python train_pipe.py --m ica --acoustic 256 260 --gpu 0 --encoder_type gsru --decoder_type lstm \
--S2ADRM_type 2branch --c3d_feats_name msrvtt_c3d_60_fc6.hdf5 --concat_before_att --n_frames 10 \
--random_type segment_random --use_preEncoder --preEncoder_modality c --scope _RMSprop

python train_pipe.py --m ica --acoustic 256 260 --gpu 1 --encoder_type gsru --decoder_type lstm \
--S2ADRM_type 2branch --c3d_feats_name msrvtt_c3d_60_fc6.hdf5 --concat_before_att --n_frames 10 \
--equally_sampling --use_preEncoder --preEncoder_modality c --scope _RMSprop
'''