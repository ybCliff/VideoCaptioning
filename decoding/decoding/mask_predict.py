
from decoding.strategy_utils import generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens
import models.Constants as Constants
import torch
from tqdm import tqdm
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from matplotlib import cm 
import torch.nn.functional as F


def plot(tgt_tokens, tgt_vocab, token_probs, corresponding_probs, num_mask, mask_ind, counter, teacher_model, split=False):
    for n in range(1,2):
        sent = []
        stu = []
        tea = []
        overall = []
        
        select_id = n
        tmp = tgt_tokens[select_id].tolist()
        mask_id = [0] * len(tmp)
        for i, token in enumerate(tmp):
            if token == Constants.PAD:
                break
            word = tgt_vocab[str(token)]
            tmp1 = token_probs[select_id, i]
            tmp2 = corresponding_probs[select_id, i]
            #jud = 'True' if tgt_tokens[select_id, i].item() == Constants.MASK else 'False'
            #tqdm.write('%s\t%.4f\t%.4f\t%.8f\t%s' % (word, tmp1, tmp2, tmp1 * tmp2, jud))
            sent.append('%s' % word)
            stu.append("%.2f" %tmp1)
            tea.append("%.2f" %tmp2)
            overall.append('%.2f' % (math.sqrt(tmp1 * tmp2)))
            if i < num_mask[select_id].item():
                mask_id[mask_ind[select_id, i].item()] = 1.0

        sent.append(str(num_mask[select_id].item()))
        tqdm.write(("Step %d: " % (counter)) + ' '.join(sent))
        tqdm.write(("Step %d Stu: " % (counter)) + ','.join(stu))
        tqdm.write(("Step %d Tea: " % (counter)) + ','.join(tea))
        tqdm.write(("Step %d All: " % (counter)) + ','.join(overall))
        mask_id = ['%.2f' % item for item in mask_id]
        tqdm.write(("Step %d Mas: " % (counter)) + ','.join(mask_id))

        stu = [float(item) for item in stu]
        tea = [float(item) for item in tea]
        overall = [float(item) for item in overall]

        if teacher_model is not None:
            a = np.array([stu[:-1], tea[:-1], overall[:-1]])
        else:
            a = np.array([stu[:-1]])


        myplot = plt.imshow(a, cmap=cm.Blues, vmin=0, vmax=1)
        cbar = plt.colorbar(myplot, shrink=.92, orientation='horizontal')

        plt.xticks(())
        plt.yticks(())
        plt.savefig('./%d_%d.png' % (1 if teacher_model is not None else 0, counter))
        plt.show()
    if split:
        tqdm.write('-----------------------')

def to_sentence(hyp, vocab, break_words=[Constants.PAD], skip_words=[]):
    sent = []
    for word_id in hyp:
        if word_id in skip_words:
            continue
        if word_id in break_words:
            break
        sent.append(vocab[str(word_id)])
    return ' '.join(sent)

class MaskPredict(object):
    
    def __init__(self, iterations, seed, dict_mapping, plot=False, collect_best_candidate_iterative_results=False):
        super().__init__()
        self.iterations = iterations
        self.random = np.random.RandomState(seed)
        self.dict_mapping = dict_mapping
        self.plot = plot
        self.collect_best_candidate_iterative_results = collect_best_candidate_iterative_results
    
    def generate(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        collect_results = []
        
        iterations = seq_len if self.iterations is None else self.iterations
        
        tgt_tokens, token_probs, all_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens)
        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens)#, no_masking_desicion=True)
        
        tgt_tokens[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0
        corresponding_probs[pad_mask] = 1.0

        if self.collect_best_candidate_iterative_results:
            collect_results.append(tgt_tokens.clone())

        #tqdm.write("Initialization: " + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))
        for counter in range(1, iterations):
            ratio = (1.0 - (counter / iterations))
            ratio = max(ratio, 0.4)

            # Mask
            num_mask = (seq_lens.float() * ratio).long()
            mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)
            if self.plot: plot(tgt_tokens, tgt_vocab, token_probs, corresponding_probs, num_mask, mask_ind, counter, teacher_model, split=True)
            tgt_tokens[mask_ind] = Constants.MASK
            
            # Predict
            new_tgt_tokens, new_token_probs, all_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens)
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]

            # Interact
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens)
            corresponding_probs[pad_mask] = 1.0

            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())

        
        if self.plot:
            plot(tgt_tokens, tgt_vocab, token_probs, corresponding_probs, num_mask, mask_ind, counter+1, teacher_model, split=True)

        #lprobs = token_probs.log()
        lprobs = (token_probs * corresponding_probs).log()
        #eos_mask = tgt_tokens.eq(Constants.EOS)
        #non_pad_eos_mask = 1 - (eos_mask + pad_mask).gt(0)
        #lengths = non_pad_eos_mask.sum(-1)


        return tgt_tokens, lprobs, collect_results
    
    def generate_non_autoregressive(self, model, enc_output, category, tgt_tokens):
        #print(enc_output[0])
        decoder_out, *_ = model.decoder(tgt_tokens, enc_output, category)
        if isinstance(decoder_out, list):
            decoder_out = decoder_out[-1]
        tgt_tokens, token_probs, all_probs = generate_step_with_prob(model.tgt_word_prj(decoder_out))
        return tgt_tokens, token_probs, all_probs

    def mapping(self, tgt_tokens):
        tokens = tgt_tokens.clone().flatten()
        for i, token in enumerate(tokens):
            tokens[i] = self.dict_mapping[token.item()]
        return tokens.view(*tgt_tokens.shape)

    def scoring_by_teacher(self, teacher_model, teacher_enc_output, category, tgt_tokens, no_masking_desicion=False):
        if teacher_model is None or no_masking_desicion:
            return tgt_tokens.new(*tgt_tokens.shape).fill_(1).float()
        if self.dict_mapping != {}:
            tokens = self.mapping(tgt_tokens)
        else:
            tokens = tgt_tokens

        tgt_tokens_with_bos = torch.cat([tokens.new(tokens.size(0), 1).fill_(Constants.BOS), tokens], dim=1)
        #print(tgt_tokens_with_bos.shape, teacher_enc_output.shape, category.shape)
        decoder_out, *_ = teacher_model.decoder(tgt_tokens_with_bos[:, :-1], teacher_enc_output, category)
        if isinstance(decoder_out, list):
            decoder_out = decoder_out[-1]
        probs = F.softmax(teacher_model.tgt_word_prj(decoder_out), dim=-1)
        return probs.gather(2, tokens.unsqueeze(2)).squeeze(2)

    '''
    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)
    '''

    def select_worst(self, token_probs, num_mask):
        masks = torch.zeros(*token_probs.shape, device=token_probs.device)
        for i in range(masks.size(0)):
            ind = token_probs[i, :].topk(max(1, num_mask[i]), largest=False, sorted=False)[1]
            masks[i, ind] = 1
        return masks.byte()
    

    def select_random(self, token_probs, num_mask, seq_lens):
        bsz, seq_len = token_probs.size()
        masks = []
        for i in range(bsz):
            ind = self.random.choice(seq_lens[i].item(), size=max(1, num_mask[i].item()), replace=False)
            ind = list(ind)
            ind += [ind[0]] * (seq_len - len(ind))
            masks.append(torch.LongTensor(ind))
        return torch.stack(masks, dim=0).to(token_probs.device)

    def select_multinomial(self, token_probs, num_mask, seq_lens):
        probs = torch.exp(-token_probs)
        bsz, seq_len = token_probs.size()
        masks = []
        for i in range(bsz):
            ind = probs[i, :int(seq_lens[i])].multinomial(max(1, num_mask[i].item()))
            ind = list(ind)
            ind += [ind[0]] * (seq_len - len(ind))
            masks.append(torch.LongTensor(ind))
        return torch.stack(masks, dim=0).to(token_probs.device)

def generate(model, teacher_model, encoder_outputs, teacher_encoder_outputs, category, tgt_tokens, tgt_vocab, opt, dict_mapping, length_bias):
    strategy = MaskPredict(opt['iterations'], opt['seed'], dict_mapping=dict_mapping)
    length_beam_size = opt['length_beam_size']
    #gold_target_len = tgt_tokens.ne(Constants.PAD).sum(-1)
    gold_target_len = None
    #gold_target_len = tgt_tokens.ne(Constants.PAD).sum(-1) if opt['use_gold_target_len'] else None
    beam_alpha = opt.get('beam_alpha', 1.0)
    #print(beam_alpha)

    enc_output, pred_length = encoder_outputs['enc_output'], encoder_outputs['pred_length']
    if teacher_encoder_outputs is not None:
        teacher_enc_output = teacher_encoder_outputs['enc_output']
        if isinstance(teacher_enc_output, list):
            teacher_enc_output = teacher_enc_output[0]
    else:
        teacher_enc_output = None
    if isinstance(enc_output, list):
        assert len(enc_output) == 1
        enc_output = enc_output[0]
    bsz = enc_output.size(0)

    beam = predict_length_beam(gold_target_len, pred_length, length_beam_size, length_bias)    
    max_len = beam.max().item()

    length_mask = torch.triu(enc_output.new(max_len, max_len).fill_(1).long(), 1)
    
    length_mask = torch.stack([length_mask[beam[batch] - 1] for batch in range(bsz)], dim=0)
    tgt_tokens = enc_output.new(bsz, length_beam_size, max_len).fill_(Constants.MASK).long()
    tgt_tokens = (1 - length_mask) * tgt_tokens + length_mask * Constants.PAD
    tgt_tokens = tgt_tokens.view(bsz * length_beam_size, max_len)
    
    enc_output = enlarge(enc_output, length_beam_size)
    category = enlarge(category, length_beam_size)
    if teacher_enc_output is not None:
        teacher_enc_output = enlarge(teacher_enc_output, length_beam_size)

    hypotheses, lprobs, collect_results = strategy.generate(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab)
    
    tgt_lengths = (1 - length_mask).sum(-1) -1
    hypotheses = hypotheses.view(bsz, length_beam_size, max_len)
    lprobs = lprobs.view(bsz, length_beam_size, max_len)
    tgt_lengths = tgt_lengths.view(bsz, length_beam_size)
    #tgt_lengths = (1 - length_mask).sum(-1)-1

    avg_log_prob = lprobs.sum(-1) / (tgt_lengths.float() ** beam_alpha)
    best_lengths = avg_log_prob.max(-1)[1]                                          # [batch_size]

    best_lengths = best_lengths.unsqueeze(1).unsqueeze(2).repeat(1, 1, max_len)     # [batch_size, 1, max_len]
    
    hypotheses = hypotheses.gather(1, best_lengths).squeeze(1)                      # [batch_size, max_len]
    #lprobs = lprobs.gather(1, best_lengths).squeeze(1)                             = [batch_size, max_len]
    lprobs = None # For speedup
    if collect_results:
        collect_results = [item.view(bsz, length_beam_size, max_len) for item in collect_results]
        #print(collect_results[0][0])
        #print(collect_results[1][0])
        #print(collect_results[2][0])
        collect_results = [item.gather(1, best_lengths).squeeze(1) for item in collect_results]
        lprobs = torch.stack(collect_results, dim=1)
    return hypotheses, lprobs


    hypotheses = torch.stack([hypotheses[b, l, :] for b, l in enumerate(best_lengths)], dim=0)
    lprobs = torch.stack([lprobs[b, l, :] for b, l in enumerate(best_lengths)], dim=0)

    
    return hypotheses, lprobs

def enlarge(info, beam_size):
    bsz, *rest_shape = info.shape
    if len(rest_shape) == 2:
        return info.unsqueeze(1).repeat(1, beam_size, 1, 1).view(bsz * beam_size, *rest_shape)
    return info.unsqueeze(1).repeat(1, beam_size, 1).view(bsz * beam_size, *rest_shape)

def predict_length_beam(gold_target_len, predicted_lengths, length_beam_size, length_bias):
    if gold_target_len is not None:
        beam_starts = gold_target_len - (length_beam_size - 1) // 2
        beam_ends = gold_target_len + length_beam_size // 2 + 1
        #beam = torch.stack([torch.arange(7, 12, device=beam_starts.device) for batch in range(gold_target_len.size(0))], dim=0)
        beam = torch.stack([torch.arange(beam_starts[batch], beam_ends[batch], device=beam_starts.device) for batch in range(gold_target_len.size(0))], dim=0)
    else:
        beam = predicted_lengths.topk(length_beam_size, dim=1)[1] + length_bias + 1
    beam[beam < 4] = 4
    beam[beam > 19] = 19
    #print(beam)
    return beam