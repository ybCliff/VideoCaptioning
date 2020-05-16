
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

def enlarge(info, beam_size, return_view=True):
    bsz, *rest_shape = info.shape
    if len(rest_shape) == 2:
        tmp = info.unsqueeze(1).repeat(1, beam_size, 1, 1)
        return tmp.view(bsz * beam_size, *rest_shape) if return_view else tmp

    tmp = info.unsqueeze(1).repeat(1, beam_size, 1)
    return tmp.view(bsz * beam_size, *rest_shape) if return_view else tmp

def to_sentence(hyp, vocab, break_words=[Constants.PAD], skip_words=[]):
    sent = []
    for word_id in hyp:
        if word_id in skip_words:
            continue
        if word_id in break_words:
            break
        sent.append(vocab[word_id])
    return ' '.join(sent)

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


'''
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

    #def select_worst(self, token_probs, num_mask):
    #    bsz, seq_len = token_probs.size()
    #    masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
    #    masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
    #    return torch.stack(masks, dim=0)
    

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

'''
class MaskPredict(object):
    
    def __init__(self, iterations, seed, dict_mapping, plot=False, collect_best_candidate_iterative_results=False, **kwargs):
        super().__init__()
        self.iterations = iterations
        self.random = np.random.RandomState(seed)
        self.dict_mapping = dict_mapping
        self.plot = plot
        self.collect_best_candidate_iterative_results = kwargs['opt'].get('collect_best_candidate_iterative_results', False)
        opt = kwargs['opt']
        self.paradigm = opt.get('paradigm', 'mp') # 'mp', 'l2r', 'r2l', 'lr2m'
        self.masking_decision = opt.get('masking_decision', False)
        self.no_candidate_decision = opt.get('no_candidate_decision', False)
    
    def generate_mp(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        collect_results = []
        collect_scores = []
        
        iterations = seq_len if self.iterations is None else self.iterations
        
        tgt_tokens, token_probs, all_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, tags)
        

        if self.collect_best_candidate_iterative_results:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        #tqdm.write("Iteration 0: " + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))
        for counter in range(1, iterations):
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=self.masking_decision)
            corresponding_probs[pad_mask] = 1.0
            ratio = (1.0 - (counter / iterations))
            #ratio = max(ratio, 0.4)

            # Mask
            num_mask = (seq_lens.float() * ratio).long()
            mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)
            if self.plot: plot(tgt_tokens, tgt_vocab, token_probs, corresponding_probs, num_mask, mask_ind, counter, teacher_model, split=True)
            tgt_tokens[mask_ind] = Constants.MASK
            
            # Predict
            new_tgt_tokens, new_token_probs, all_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, tags)
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]

            # Interact
            

            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())

            #tqdm.write(("Iteration %d: " % counter) + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))

        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        
        if self.plot:
            plot(tgt_tokens, tgt_vocab, token_probs, corresponding_probs, num_mask, mask_ind, counter+1, teacher_model, split=True)

        #lprobs = token_probs.log()
        lprobs = (token_probs * corresponding_probs).log()
        #eos_mask = tgt_tokens.eq(Constants.EOS)
        #non_pad_eos_mask = 1 - (eos_mask + pad_mask).gt(0)
        #lengths = non_pad_eos_mask.sum(-1)


        return tgt_tokens, lprobs, (collect_results, collect_scores), None
    
    def generate_sequential(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags, direction, step=1):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        non_pad_mask = tgt_tokens.ne(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        collect_results = []
        collect_scores = []
        
        token_probs = tgt_tokens.new(*tgt_tokens.shape).fill_(0).float()
        token_probs[pad_mask] = 1.0

        if self.collect_best_candidate_iterative_results:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        itrs = [i for i in range(0, seq_len, step)] if direction == 0 else [i for i in range(seq_len-1, -1, -step)]
        for counter in itrs:
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=self.masking_decision)
            corresponding_probs[pad_mask] = 1.0
            masks = torch.zeros(*token_probs.shape, device=token_probs.device)
            if direction == 0:
                masks[:, counter:min(counter+step,seq_len)] = 1
            else:
                masks[:, max(counter-step, 0):counter] = 1

            mask_ind = masks.byte() & non_pad_mask
            #print(mask_ind[1].tolist())
            tgt_tokens[mask_ind] = Constants.MASK

            tqdm.write(("Iteration %d1 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            new_tgt_tokens, new_token_probs, _ = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, tags)
            # Predict
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            
            tqdm.write(("Iteration %d2 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())


        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()

        return tgt_tokens, lprobs, (collect_results, collect_scores), None#visual_mask.sum(-1)
        #lprobs = token_probs.log()
        lprobs = (token_probs * corresponding_probs).log()
        #eos_mask = tgt_tokens.eq(Constants.EOS)
        #non_pad_eos_mask = 1 - (eos_mask + pad_mask).gt(0)
        #lengths = non_pad_eos_mask.sum(-1)


        return tgt_tokens, lprobs, (collect_results, collect_scores), None
    

    def generate(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags):
        if self.paradigm == 'mp':
            return self.generate_mp(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags)
        elif self.paradigm == 'l2r':
            return self.generate_sequential(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags, direction=0)
        elif self.paradigm == 'r2l':
            return self.generate_sequential(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags, direction=1)


    def generate_non_autoregressive(self, model, enc_output, category, tgt_tokens, pad_mask, tags):
        #print(enc_output[0])
        decoder_out, *_ = model.decoder(tgt_tokens, enc_output, category, tags=tags)
        if isinstance(decoder_out, list):
            decoder_out = decoder_out[-1]
        tgt_tokens, token_probs, all_probs = generate_step_with_prob(model.tgt_word_prj(decoder_out))
        tgt_tokens[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0
        return tgt_tokens, token_probs, all_probs

    def mapping(self, tgt_tokens):
        tokens = tgt_tokens.clone().flatten()
        for i, token in enumerate(tokens):
            tokens[i] = self.dict_mapping[token.item()]
        return tokens.view(*tgt_tokens.shape)

    def scoring_by_teacher(self, teacher_model, teacher_enc_output, category, tgt_tokens, decision=True):
        if teacher_model is None or not decision:
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

def generate(model, teacher_model, encoder_outputs, teacher_encoder_outputs, category, tgt_tokens, tgt_vocab, opt, dict_mapping, length_bias, tags):
    if opt['method'] == 'mp':
        func = MaskPredict
    elif opt['method'] == 'direct':
        func = NVA
    elif opt['method'] == 'ap':
        func = AllPredict
    elif opt['method'] == 'signal' or opt['method'] == 'signal2':
        func = Signal
    elif opt['method'] == 'signal3':
        func = Signal3
    elif opt['method'] == 'nv':
        func = NV
    elif opt['method'] == 'ms':
        func = MS

    strategy = func(opt['iterations'], opt['seed'], dict_mapping=dict_mapping, masking_ratio=opt['masking_ratio'], opt=opt)

    length_beam_size = opt['length_beam_size']
    if opt.get('load_generated_captions', False):
        gold_target_len = tgt_tokens.ne(Constants.PAD).sum(-1)
    else:
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
    if gold_target_len is not None:
        tgt_tokens = tgt_tokens[:, :max_len]
        tgt_tokens[tgt_tokens==Constants.PAD] = Constants.MASK
        tgt_tokens = tgt_tokens.unsqueeze(1).repeat(1, length_beam_size, 1)
    else:
        tgt_tokens = enc_output.new(bsz, length_beam_size, max_len).fill_(Constants.MASK if not opt.get('use_eos',False) else Constants.EOS).long()
    tgt_tokens = (1 - length_mask) * tgt_tokens + length_mask * Constants.PAD
    #print(tgt_tokens[0])
    tgt_tokens = tgt_tokens.view(bsz * length_beam_size, max_len)
    
    enc_output = enlarge(enc_output, length_beam_size)
    category = enlarge(category, length_beam_size)
    if tags is not None:
        tags = enlarge(tags, length_beam_size)
    if teacher_enc_output is not None:
        teacher_enc_output = enlarge(teacher_enc_output, length_beam_size)

    hypotheses, lprobs, collect_results, visual_mask = strategy.generate(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags)
    
    if visual_mask is not None:
        visual_mask = visual_mask.view(bsz, length_beam_size).long()
        tgt_lengths = (1 - length_mask).sum(-1) - visual_mask
    else:
        tgt_lengths = (1 - length_mask).sum(-1)
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
    assert isinstance(collect_results, tuple)
    if collect_results[0]:
        sents, scores = collect_results
        if not opt.get('not_only_best_candidate', False) and not opt.get('collect_last', False):
            sents = [item.view(bsz, length_beam_size, max_len) for item in sents]
            sents = [item.gather(1, best_lengths).squeeze(1) for item in sents]

            scores = [item.view(bsz, length_beam_size, max_len) for item in scores]
            scores = [item.gather(1, best_lengths).squeeze(1) for item in scores]

        lprobs = (torch.stack(sents, dim=1), torch.stack(scores, dim=1))
    return hypotheses, lprobs


    hypotheses = torch.stack([hypotheses[b, l, :] for b, l in enumerate(best_lengths)], dim=0)
    lprobs = torch.stack([lprobs[b, l, :] for b, l in enumerate(best_lengths)], dim=0)

    
    return hypotheses, lprobs


def predict_length_beam(gold_target_len, predicted_lengths, length_beam_size, length_bias):
    if gold_target_len is not None:
        beam_starts = gold_target_len - (length_beam_size - 1) // 2
        beam_ends = gold_target_len + length_beam_size // 2 + 1
        #beam = torch.stack([torch.arange(7, 12, device=beam_starts.device) for batch in range(gold_target_len.size(0))], dim=0)
        beam = torch.stack([torch.arange(beam_starts[batch], beam_ends[batch], device=beam_starts.device) for batch in range(gold_target_len.size(0))], dim=0)
    else:
        beam = predicted_lengths.topk(length_beam_size, dim=1)[1] + length_bias

    beam[beam < 4] = 4
    beam[beam > 19] = 19
    #print(beam)
    return beam

'''
class NVA(object):
    
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
        
        tgt_tokens, token_probs, all_probs = self.generate_non_autoregressive(model.decoder.decoder1, enc_output, category, tgt_tokens, model.tgt_word_prj)
        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens)#, no_masking_desicion=True)
        
        tgt_tokens[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0
        corresponding_probs[pad_mask] = 1.0

        if self.collect_best_candidate_iterative_results:
            collect_results.append(tgt_tokens.clone())

        tqdm.write("Iteration 0: " + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))
        for counter in range(1, iterations):
            ratio = (1.0 - (counter / iterations))
            ratio = max(ratio, 0.4)

            # Mask
            if counter == 1:
                mask_ind = tgt_tokens.eq(Constants.MASK)
            else:
                num_mask = (seq_lens.float() * ratio).long()
                mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)
                if self.plot: plot(tgt_tokens, tgt_vocab, token_probs, corresponding_probs, num_mask, mask_ind, counter, teacher_model, split=True)
                tgt_tokens[mask_ind] = Constants.MASK
            
            # Predict
            new_tgt_tokens, new_token_probs, all_probs = self.generate_non_autoregressive(model.decoder.decoder2, enc_output, category, tgt_tokens, model.tgt_word_prj)
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]

            # Interact
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens)
            corresponding_probs[pad_mask] = 1.0

            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())

            tqdm.write(("Iteration %d: " % counter) + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))

        
        if self.plot:
            plot(tgt_tokens, tgt_vocab, token_probs, corresponding_probs, num_mask, mask_ind, counter+1, teacher_model, split=True)

        #lprobs = token_probs.log()
        lprobs = (token_probs * corresponding_probs).log()
        #eos_mask = tgt_tokens.eq(Constants.EOS)
        #non_pad_eos_mask = 1 - (eos_mask + pad_mask).gt(0)
        #lengths = non_pad_eos_mask.sum(-1)


        return tgt_tokens, lprobs, collect_results
    
    def generate_non_autoregressive(self, decoder, enc_output, category, tgt_tokens, tgt_word_prj):
        #print(enc_output[0])
        decoder_out, *_ = decoder(tgt_tokens, enc_output, category)
        if isinstance(decoder_out, list):
            decoder_out = decoder_out[-1]
        tgt_tokens, token_probs, all_probs = generate_step_with_prob(tgt_word_prj(decoder_out))
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
'''
class NVA(object):
    
    def __init__(self, iterations, seed, dict_mapping, plot=False, collect_best_candidate_iterative_results=False, **kwargs):
        super().__init__()
        self.iterations = iterations
        self.random = np.random.RandomState(seed)
        self.dict_mapping = dict_mapping
        self.plot = plot
        self.collect_best_candidate_iterative_results = collect_best_candidate_iterative_results
        self.masking_ratio = kwargs['masking_ratio']
    
    def generate(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        collect_results = []
        
        iterations = self.iterations
        
        tgt_tokens, token_probs, all_probs = self.generate_non_autoregressive(model.decoder.decoder1, enc_output, category, tgt_tokens, model.tgt_word_prj)
        tgt_tokens[pad_mask] = Constants.PAD


        #tqdm.write("Iteration 0: " + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))
        if iterations > 1:
            tgt_tokens, token_probs, all_probs = self.generate_non_autoregressive(model.decoder.decoder2, enc_output, category, tgt_tokens, model.tgt_word_prj)
            tgt_tokens[pad_mask] = Constants.PAD
            #tqdm.write("Iteration 1: " + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))
            token_probs[pad_mask] = 1.0
        
        for counter in range(2, iterations):
            ratio = (1.0 - (counter / iterations))

            # Mask
            num_mask = (seq_lens.float() * ratio).long()
            mask_ind = self.select_worst(token_probs, num_mask)
            tgt_tokens[mask_ind] = Constants.MASK
            
            # Predict
            new_tgt_tokens, new_token_probs, all_probs = self.generate_non_autoregressive(model.decoder.decoder2, enc_output, category, tgt_tokens, model.tgt_word_prj)
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]


            #tqdm.write(("Iteration %d: " % counter) + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))

        
        lprobs = (token_probs).log()
        return tgt_tokens, lprobs, collect_results
    
    def generate_non_autoregressive(self, decoder, enc_output, category, tgt_tokens, tgt_word_prj, zeros=[]):
        #print(enc_output[0])
        decoder_out, *_ = decoder(tgt_tokens, enc_output, category)
        if isinstance(decoder_out, list):
            decoder_out = decoder_out[-1]
        tgt_tokens, token_probs, all_probs = generate_step_with_prob(tgt_word_prj(decoder_out), zeros=zeros)
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



class AllPredict(object):
    
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
        non_pad_mask = tgt_tokens.ne(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        collect_results = []
        
        iterations = seq_len if self.iterations is None else self.iterations
        
        tgt_tokens, token_probs, all_probs = self.generate_non_autoregressive(model.decoder.decoder1, enc_output, category, tgt_tokens, model.tgt_word_prj, zeros=[Constants.MASK])
        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens)#, no_masking_desicion=True)
        
        tgt_tokens[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0
        corresponding_probs[pad_mask] = 1.0

        if self.collect_best_candidate_iterative_results:
            collect_results.append(tgt_tokens.clone())

        tqdm.write("Iteration 0: " + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))
        for counter in range(1, iterations):
            
            # Predict
            new_tgt_tokens, new_token_probs, all_probs = self.generate_non_autoregressive(model.decoder.decoder2, enc_output, category, tgt_tokens, model.tgt_word_prj)
            token_probs[non_pad_mask] = new_token_probs[non_pad_mask]
            tgt_tokens[non_pad_mask] = new_tgt_tokens[non_pad_mask]

            # Interact
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens)
            corresponding_probs[pad_mask] = 1.0

            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())

            tqdm.write(("Iteration %d: " % counter) + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))

        
        if self.plot:
            plot(tgt_tokens, tgt_vocab, token_probs, corresponding_probs, num_mask, mask_ind, counter+1, teacher_model, split=True)

        #lprobs = token_probs.log()
        lprobs = (token_probs * corresponding_probs).log()
        #eos_mask = tgt_tokens.eq(Constants.EOS)
        #non_pad_eos_mask = 1 - (eos_mask + pad_mask).gt(0)
        #lengths = non_pad_eos_mask.sum(-1)


        return tgt_tokens, lprobs, collect_results
    
    def generate_non_autoregressive(self, decoder, enc_output, category, tgt_tokens, tgt_word_prj, zeros=[]):
        #print(enc_output[0])
        decoder_out, *_ = decoder(tgt_tokens, enc_output, category)
        if isinstance(decoder_out, list):
            decoder_out = decoder_out[-1]
        tgt_tokens, token_probs, all_probs = generate_step_with_prob(tgt_word_prj(decoder_out), zeros=zeros)
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



class Signal(object):
    
    def __init__(self, iterations, seed, dict_mapping, plot=False, collect_best_candidate_iterative_results=False, **kwargs):
        super().__init__()
        self.iterations = iterations
        self.random = np.random.RandomState(seed)
        self.dict_mapping = dict_mapping
        self.plot = plot
        self.collect_best_candidate_iterative_results = collect_best_candidate_iterative_results
        self.masking_ratio = kwargs['masking_ratio']
    
    def generate(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        collect_results = []
        
        iterations = self.iterations
        #tqdm.write("Initilazation: " + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))
        tgt_tokens, token_probs, all_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, signal=0)
        tgt_tokens[pad_mask] = Constants.PAD


        tqdm.write("Iteration 0: " + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))
        if iterations > 1:
            tgt_tokens, token_probs, all_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, signal=1)
            tgt_tokens[pad_mask] = Constants.PAD
            tqdm.write("Iteration 1: " + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))
            token_probs[pad_mask] = 1.0
        
        for counter in range(2, iterations):
            ratio = (1.0 - (counter / iterations))
            #ratio = max(ratio, 0.4)

            # Mask
            num_mask = (seq_lens.float() * ratio).long()
            mask_ind = self.select_worst(token_probs, num_mask)
            tgt_tokens[mask_ind] = Constants.MASK
            tqdm.write(("Iteration %d: " % counter) + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))
            # Predict
            new_tgt_tokens, new_token_probs, all_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, signal=1)
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]


            tqdm.write(("Iteration %d: " % counter) + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))

        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens)
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()
        
        #lprobs = (token_probs).log()
        return tgt_tokens, lprobs, collect_results
    
    def generate_non_autoregressive(self, model, enc_output, category, tgt_tokens, signal, zeros=[]):
        decoder_out = model.decoder.forward_(tgt_tokens, enc_output, category, signal=signal)
        tgt_tokens, token_probs, all_probs = generate_step_with_prob(model.tgt_word_prj(decoder_out), zeros=zeros)
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

'''
class Signal3(object):
    
    def __init__(self, iterations, seed, dict_mapping, plot=False, collect_best_candidate_iterative_results=False, **kwargs):
        super().__init__()
        self.iterations = iterations
        self.random = np.random.RandomState(seed)
        self.dict_mapping = dict_mapping
        self.plot = plot
        self.collect_best_candidate_iterative_results = collect_best_candidate_iterative_results
        opt = kwargs['opt']
        self.visual_tag = opt['visual_tag']
        self.nonvisual_tag = opt['nonvisual_tag']
        self.revision_tag = opt['revision_tag']


    def separation_integration(self, model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab):
        mask_ind = tgt_tokens.eq(Constants.MASK)
        t1, t2 = tgt_tokens.clone(), tgt_tokens.clone()
        t1[mask_ind], t2[mask_ind] = self.visual_tag, self.nonvisual_tag

        t1, t1_probs, copy1 = self.generate_non_autoregressive(model, enc_output, category, t1, pad_mask, signal=0, tag_replace=[self.visual_tag, self.revision_tag])
        tqdm.write("    Visual   : " + to_sentence(t1[0].tolist(), tgt_vocab))
        t2, t2_probs, copy2 = self.generate_non_autoregressive(model, enc_output, category, t2, pad_mask, signal=1, tag_replace=[self.nonvisual_tag, self.revision_tag])
        tqdm.write("  Non Visual : " + to_sentence(t2[0].tolist(), tgt_vocab))

        ind_blank = t1.eq(self.visual_tag) & t2.eq(self.nonvisual_tag)
        ind = t2_probs > t1_probs
        t1[ind] = t2[ind]
        t1_probs[ind] = t2_probs[ind]
        t1_probs[ind_blank] = torch.max(copy1[ind_blank], copy2[ind_blank])

        tqdm.write("    Fusion   : " + to_sentence(t1[0].tolist(), tgt_vocab))

        return t1, t1_probs

    def generate(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        collect_results = []
        
        iterations = self.iterations
        
        tgt_tokens, token_probs = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab)

        for counter in range(1, iterations):
            ratio = (1.0 - (counter / iterations))
            num_mask = (seq_lens.float() * ratio).long()
            mask_ind = self.select_worst(token_probs, num_mask)
            tgt_tokens[mask_ind] = Constants.MASK

            # Predict
            tgt_tokens, token_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, signal=2)

            tqdm.write(("Iteration %d: " % counter) + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))

        
        lprobs = (token_probs).log()
        return tgt_tokens, lprobs, collect_results
    
    def generate_non_autoregressive(self, model, enc_output, category, tgt_tokens, pad_mask, signal, zeros=[], tag_replace=None):
        decoder_out = model.decoder.forward_(tgt_tokens, enc_output, category, signal=signal)
        tgt_tokens, token_probs, all_probs = generate_step_with_prob(model.tgt_word_prj(decoder_out), zeros=zeros)

        tgt_tokens[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0

        if tag_replace is not None:
            source, target = tag_replace
            ind = tgt_tokens.eq(source)
            tgt_tokens[ind] = target
            copy_ = token_probs.clone()
            token_probs[ind] = 0.0
            return tgt_tokens, token_probs, copy_
        return tgt_tokens, token_probs

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


class Signal3(object):
    def __init__(self, iterations, seed, dict_mapping, plot=False, collect_best_candidate_iterative_results=False, **kwargs):
        super().__init__()
        self.iterations = iterations
        self.random = np.random.RandomState(seed)
        self.dict_mapping = dict_mapping
        self.plot = plot
        self.collect_best_candidate_iterative_results = collect_best_candidate_iterative_results
        opt = kwargs['opt']
        self.visual_tag = opt['visual_tag']
        self.nonvisual_tag = opt['nonvisual_tag']
        self.revision_tag = opt['revision_tag']


    def separation_integration(self, model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab):
        mask_ind = tgt_tokens.eq(Constants.MASK)
        t1, t2 = tgt_tokens.clone(), tgt_tokens.clone()
        t1[mask_ind], t2[mask_ind] = self.visual_tag, self.nonvisual_tag

        t1, t1_probs, copy1 = self.generate_non_autoregressive(model, enc_output, category, t1, pad_mask, signal=0, tag_replace=[self.revision_tag, self.revision_tag])
        tqdm.write("    Visual   : " + to_sentence(t1[0].tolist(), tgt_vocab))
        t2, t2_probs, copy2 = self.generate_non_autoregressive(model, enc_output, category, t2, pad_mask, signal=1, tag_replace=[self.revision_tag, self.revision_tag])
        tqdm.write("  Non Visual : " + to_sentence(t2[0].tolist(), tgt_vocab))

        ind_blank = t1.eq(self.revision_tag) & t2.eq(self.revision_tag)
        
        ind = t2_probs > t1_probs
        t1[ind] = t2[ind]
        t1_probs[ind] = t2_probs[ind]
        t1_probs[ind_blank] = 0.0

        tqdm.write("    Fusion   : " + to_sentence(t1[0].tolist(), tgt_vocab))

        return t1, t1_probs

    def generate(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        collect_results = []
        
        iterations = self.iterations
        
        tgt_tokens, token_probs = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab)

        for counter in range(1, iterations):
            ratio = (1.0 - (counter / iterations))
            num_mask = (seq_lens.float() * ratio).long()
            mask_ind = self.select_worst(token_probs, num_mask)
            tgt_tokens[mask_ind] = Constants.MASK
            tqdm.write(("Iteration %d_0: " % counter) + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))
            # Predict
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, signal=2)

            

            # Predict
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            tqdm.write(("Iteration %d_1: " % counter) + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))

        
        lprobs = (token_probs).log()
        return tgt_tokens, lprobs, collect_results
    
    def generate_non_autoregressive(self, model, enc_output, category, tgt_tokens, pad_mask, signal, zeros=[], tag_replace=None):
        decoder_out = model.decoder.forward_(tgt_tokens, enc_output, category, signal=signal)
        tgt_tokens, token_probs, all_probs = generate_step_with_prob(model.tgt_word_prj(decoder_out), zeros=zeros)

        tgt_tokens[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0

        if tag_replace is not None:
            source, target = tag_replace
            ind = tgt_tokens.eq(source)
            tgt_tokens[ind] = target
            copy_ = token_probs.clone()
            token_probs[ind] = 0.0
            return tgt_tokens, token_probs, copy_
        return tgt_tokens, token_probs

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
'''

class Signal3(object):
    
    def __init__(self, iterations, seed, dict_mapping, plot=False, collect_best_candidate_iterative_results=False, **kwargs):
        super().__init__()
        self.iterations = iterations
        self.random = np.random.RandomState(seed)
        self.dict_mapping = dict_mapping
        self.plot = plot
        self.collect_best_candidate_iterative_results = collect_best_candidate_iterative_results
        opt = kwargs['opt']
        self.visual_tag = opt['visual_tag']
        self.nonvisual_tag = opt['nonvisual_tag']
        self.revision_tag = opt['revision_tag']


    def separation_integration(self, model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab):
        mask_ind = tgt_tokens.eq(Constants.MASK)
        t1, t2 = tgt_tokens.clone(), tgt_tokens.clone()
        t1[mask_ind], t2[mask_ind] = self.visual_tag, self.nonvisual_tag

        t1, t1_probs, copy1 = self.generate_non_autoregressive(model, enc_output, category, t1, pad_mask, signal=0, tag_replace=[self.visual_tag, self.revision_tag])
        tqdm.write("    Visual   : " + to_sentence(t1[0].tolist(), tgt_vocab))
        t2, t2_probs, copy2 = self.generate_non_autoregressive(model, enc_output, category, t2, pad_mask, signal=0, tag_replace=[self.nonvisual_tag, self.revision_tag])
        tqdm.write("  Non Visual : " + to_sentence(t2[0].tolist(), tgt_vocab))

        ind_blank = t1.eq(self.visual_tag) & t2.eq(self.nonvisual_tag)
        ind = t2_probs > t1_probs
        t1[ind] = t2[ind]
        t1_probs[ind] = t2_probs[ind]
        t1_probs[ind_blank] = 0.0 #torch.max(copy1[ind_blank], copy2[ind_blank])

        tqdm.write("    Fusion   : " + to_sentence(t1[0].tolist(), tgt_vocab))

        return t1, t1_probs

    def generate(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        collect_results = []
        
        iterations = self.iterations
        
        tgt_tokens, token_probs = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab)

        for counter in range(1, iterations):
            ratio = (1.0 - (counter / iterations))
            num_mask = (seq_lens.float() * ratio).long()
            mask_ind = self.select_worst(token_probs, num_mask)
            tgt_tokens[mask_ind] = Constants.MASK

            # Predict
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, signal=1)

            # Predict
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            tqdm.write(("Iteration %d: " % counter) + to_sentence(tgt_tokens[0].tolist(), tgt_vocab))

        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens)
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()
        
        #lprobs = (token_probs).log()
        return tgt_tokens, lprobs, collect_results
    
    def generate_non_autoregressive(self, model, enc_output, category, tgt_tokens, pad_mask, signal, zeros=[], tag_replace=None):
        decoder_out = model.decoder.forward_(tgt_tokens, enc_output, category, signal=signal)
        tgt_tokens, token_probs, all_probs = generate_step_with_prob(model.tgt_word_prj(decoder_out), zeros=zeros)

        tgt_tokens[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0

        if tag_replace is not None:
            source, target = tag_replace
            ind = tgt_tokens.eq(source)
            tgt_tokens[ind] = target
            copy_ = token_probs.clone()
            token_probs[ind] = 0.0
            return tgt_tokens, token_probs, copy_
        return tgt_tokens, token_probs

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


class NV(object):
    
    def __init__(self, iterations, seed, dict_mapping, plot=False, **kwargs):
        super().__init__()
        self.iterations = iterations
        self.random = np.random.RandomState(seed)
        self.dict_mapping = dict_mapping
        self.plot = plot
        opt = kwargs['opt']
        self.visual_tag = opt['visual_tag']
        self.nonvisual_tag = opt['nonvisual_tag']
        self.revision_tag = opt['revision_tag']
        self.masking_decision = opt.get('masking_decision', False)
        self.no_candidate_decision = opt.get('no_candidate_decision', False)
        self.collect_best_candidate_iterative_results = opt.get('collect_best_candidate_iterative_results', False)
        self.collect_last = opt.get('collect_last', False)

        self.scale = opt.get('nv_scale', 0.0)
        self.fixed_iterations = opt.get('fixed_iterations', -1)
        self.load_generated_captions = opt.get('load_generated_captions', False)

        if self.fixed_iterations != -1: assert self.scale > 0
        #assert self.fixed_iterations <= self.iterations - 2

        self.paradigm = opt.get('paradigm', 'mp') # 'mp', 'l2r', 'r2l', 'lr2m'
        self.q = opt.get('q', 1)


    def separation_integration(self, model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab, tags):
        if self.load_generated_captions:
            new_tgt_tokens, new_token_probs, all_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, tags, signal=1, return_all_probs=True)
            token_probs = all_probs.gather(2, tgt_tokens.unsqueeze(2)).squeeze(2) / 3
            token_probs[pad_mask] = 1.0
            return tgt_tokens, token_probs, None


        mask_ind = tgt_tokens.eq(Constants.MASK)
        t1, t2 = tgt_tokens.clone(), tgt_tokens
        t1[mask_ind] = self.visual_tag

        t1, t1_probs = self.generate_non_autoregressive(model, enc_output, category, t1, pad_mask, tags, signal=0)
        if self.scale == 100:
            #token_probs = tgt_tokens.new(*tgt_tokens.shape).fill_(0).float()
            #token_probs[pad_mask] = 1.0
            #return tgt_tokens, token_probs, None
            t1_probs[t1.eq(Constants.MASK)] = 0.0
            return t1, t1_probs, None
        #tqdm.write("    Visual   : " + to_sentence(t1[1].tolist(), tgt_vocab))
        #tqdm.write("    Visual   : " + ' '.join([('%.3f'%item if item!=1.0 else '') for item in t1_probs[1].tolist()]))
        t2, t2_probs = self.generate_non_autoregressive(model, enc_output, category, t2, pad_mask, tags, signal=1)
        #tqdm.write("    Mask     : " + to_sentence(t2[1].tolist(), tgt_vocab))
        #tqdm.write("    Mask     : " + ' '.join([('%.3f'%item if item!=1.0 else '') for item in t2_probs[1].tolist()]))

        if self.scale == 0:
            ind = None
        elif self.scale == 100:
            t1_probs[t1.eq(Constants.MASK)] = 0.0
            t2 = t1
            t2_probs = t1_probs
            ind = None
        elif self.scale == 1000:
            t1_probs[t1.eq(Constants.MASK)] = 0.0
            t1[pad_mask] = Constants.MASK
            ind = t1.ne(Constants.MASK)

            not_equal = (t2[ind] != t1[ind])
            tmp_t = t2[ind].clone()
            tmp_t[not_equal] = t1[ind][not_equal]
            tmp_probs = t2_probs[ind].clone()
            tmp_probs[not_equal] = 2 * t1_probs[ind][not_equal]
            print(not_equal.sum().item())
            t2[ind] = tmp_t
            t2_probs[ind] = tmp_probs
            t2_probs[t2_probs>1.0] = 1.0

            '''
            equal = (t2[ind] = t1[ind])
            tmp_probs = t2_probs[ind].clone()
            tmp_probs[equal] += t1_probs[ind][equal]
            t2[ind] = t1[ind]
            t2_probs[ind] = tmp_probs #(t2_probs[ind]+t1_probs[ind])/2 #torch.sqrt(t2_probs[ind]*t1_probs[ind])
            t2_probs[t2_probs>1.0] = 1.0
            '''
        elif self.scale == 10000:
            t1_probs[t1.eq(Constants.MASK)] = 0.0
            t2 = t1
            t2_probs = t1_probs
            ind = None
        else:
            t1_probs[t1.eq(Constants.MASK)] = 0.0
            #ind = t1_probs > t2_probs
            t1[pad_mask] = Constants.MASK
            ind = t1.ne(Constants.MASK)
            t2[ind] = t1[ind]
            #t2_probs[ind] = t1_probs[ind]
            t2_probs[ind] = self.scale*t1_probs[ind]
            t2_probs[t2_probs>1.0] = 1.0

        #tqdm.write("    Fusion   : " + to_sentence(t2[1].tolist(), tgt_vocab))

        return t2, t2_probs, ind
        #return t1, t1_probs

    def generate_mp(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags):
        collect_results = []
        collect_scores = []
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens1 = seq_len - pad_mask.sum(dim=1)
        
        iterations = self.iterations if self.scale != 100 else self.iterations + 1

        #tqdm.write(("Iteration 0 : ") + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
        
        tgt_tokens, token_probs, visual_mask = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab, tags)
        visual_probs = token_probs[visual_mask]
        seq_lens2 = seq_lens1 - visual_mask.sum(-1) if visual_mask is not None else seq_lens1
        #if visual_mask is not None:
        #    seq_lens = seq_lens - visual_mask.sum(-1)
        #print(visual_mask.long().sum(-1).float())

        if self.collect_best_candidate_iterative_results and not self.collect_last:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        for counter in range(1, iterations):
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=self.masking_decision)
            corresponding_probs[pad_mask] = 1.0
            #tqdm.write(("Iteration %dte: " % counter) + ' '.join([('%.3f'%item if item!=1.0 else '') for item in corresponding_probs[1].tolist()]))
            #tqdm.write(("Iteration %dst: " % counter) + ' '.join([('%.3f'%item if item!=1.0 else '') for item in token_probs[1].tolist()]))
            if self.fixed_iterations != -1:
                if counter - 1 != self.fixed_iterations:
                    token_probs[visual_mask] = 1.0
                    #seq_lens = seq_lens2
                    seq_lens = seq_lens1
                else:
                    token_probs[visual_mask] = visual_probs
                    seq_lens = seq_lens1
            else:
                seq_lens = seq_lens1

            if self.scale == 100:
                if counter == 1:
                    mask_ind = (tgt_tokens == Constants.MASK)
                else:
                    #ratio = max((1.0 - (counter / iterations)), 0.3)
                    ratio = (1.0 - (counter / iterations))
                    #ratio = (1.0 - ((counter-1) / (iterations-1)))
                    #ratio = 0.4
                    #ratio = min((1.0 - (counter / iterations)), 0.7)
                    num_mask = (seq_lens.float() * ratio).long()
                    mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)
                    #mask_ind = self.select_worst(token_probs, num_mask)
            else:
                #ratio = max((1.0 - (counter / iterations)), 0.2)
                ratio = (1.0 - (counter / iterations))
                #ratio = min((1.0 - (counter / iterations)), 0.7)
                if self.load_generated_captions:
                    ratio *= 0.01
                num_mask = (seq_lens.float() * ratio).long()
                mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)
                #mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            #tqdm.write(("Iteration %d1 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            # Predict
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, tags, signal=1)
            #tqdm.write(("Iteration %d0 : " % counter) + to_sentence(new_tgt_tokens[1].tolist(), tgt_vocab))
            # Predict
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            #tqdm.write(("Iteration %d2 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            if self.collect_best_candidate_iterative_results and not self.collect_last:
                collect_results.append(tgt_tokens.clone())

                if counter == iterations - 1:
                    corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
                    corresponding_probs[pad_mask] = 1.0
                    collect_scores.append((token_probs * corresponding_probs).clone())
                else:
                	collect_scores.append(token_probs.clone())


        if self.collect_last:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()
        
        #lprobs = (token_probs).log()
        return tgt_tokens, lprobs, (collect_results, collect_scores), None#visual_mask.sum(-1)

    def generate_ap(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags):
        collect_results = []
        collect_scores = []
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens1 = seq_len - pad_mask.sum(dim=1)
        
        iterations = self.iterations if self.scale != 100 else self.iterations + 1

        #tqdm.write(("Iteration 0 : ") + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
        
        tgt_tokens, token_probs, visual_mask = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab, tags)
        visual_probs = token_probs[visual_mask]
        seq_lens2 = seq_lens1 - visual_mask.sum(-1) if visual_mask is not None else seq_lens1
        #if visual_mask is not None:
        #    seq_lens = seq_lens - visual_mask.sum(-1)
        #print(visual_mask.long().sum(-1).float())

        if self.collect_best_candidate_iterative_results and not self.collect_last:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        for counter in range(1, iterations):
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=self.masking_decision)
            corresponding_probs[pad_mask] = 1.0
            #tqdm.write(("Iteration %dte: " % counter) + ' '.join([('%.3f'%item if item!=1.0 else '') for item in corresponding_probs[1].tolist()]))
            #tqdm.write(("Iteration %dst: " % counter) + ' '.join([('%.3f'%item if item!=1.0 else '') for item in token_probs[1].tolist()]))
            if self.fixed_iterations != -1:
                if counter - 1 != self.fixed_iterations:
                    token_probs[visual_mask] = 1.0
                    #seq_lens = seq_lens2
                    seq_lens = seq_lens1
                else:
                    token_probs[visual_mask] = visual_probs
                    seq_lens = seq_lens1
            else:
                seq_lens = seq_lens1

            if self.scale == 100:
                if counter == 1:
                    mask_ind = (tgt_tokens == Constants.MASK)
                else:
                    #ratio = max((1.0 - (counter / iterations)), 0.3)
                    ratio = (1.0 - (counter / iterations))
                    #ratio = (1.0 - ((counter-1) / (iterations-1)))
                    #ratio = 0.4
                    #ratio = min((1.0 - (counter / iterations)), 0.7)
                    num_mask = (seq_lens.float() * ratio).long()
                    mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)
                    #mask_ind = self.select_worst(token_probs, num_mask)
            else:
                #ratio = max((1.0 - (counter / iterations)), 0.2)
                ratio = (1.0 - (counter / iterations))
                #ratio = min((1.0 - (counter / iterations)), 0.7)
                if self.load_generated_captions:
                    ratio *= 0.01
                num_mask = (seq_lens.float() * ratio).long()
                mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)
                #mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            #tqdm.write(("Iteration %d1 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            # Predict
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, tags, signal=1)
            #tqdm.write(("Iteration %d0 : " % counter) + to_sentence(new_tgt_tokens[1].tolist(), tgt_vocab))
            # Predict
            token_probs, tgt_tokens = new_token_probs, new_tgt_tokens
            #tqdm.write(("Iteration %d2 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            if self.collect_best_candidate_iterative_results and not self.collect_last:
                collect_results.append(tgt_tokens.clone())

                if counter == iterations - 1:
                    corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
                    corresponding_probs[pad_mask] = 1.0
                    collect_scores.append((token_probs * corresponding_probs).clone())
                else:
                    collect_scores.append(token_probs.clone())


        if self.collect_last:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()
        
        #lprobs = (token_probs).log()
        return tgt_tokens, lprobs, (collect_results, collect_scores), None#visual_mask.sum(-1)

    
    def generate_sequential(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags, direction, step=1):
        collect_results = []
        collect_scores = []
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        non_pad_mask = tgt_tokens.ne(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        if self.scale == 100:
            tgt_tokens, token_probs, visual_mask = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab, tags)
            #visual_mask = tgt_tokens.ne(Constants.MASK)
        else:
            token_probs = tgt_tokens.new(*tgt_tokens.shape).fill_(0).float()
            token_probs[pad_mask] = 1.0

        if self.collect_best_candidate_iterative_results:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        
        
        def get_mask_ind(tgt_tokens, seq_lens):
            all_mask_ind = []
            for i in range(tgt_tokens.size(0)):
                item = [j for j in range(seq_lens[i]) if tgt_tokens[i, j] == Constants.MASK]
                all_mask_ind.append(item)
            return all_mask_ind

        def select_left(all_mask_ind, current, step):
            masks = torch.zeros(*token_probs.shape, device=token_probs.device)
            for i in range(masks.size(0)):
                ind = all_mask_ind[i][current:min(current+step,len(all_mask_ind[i]))] if current < len(all_mask_ind[i]) else []
                masks[i, ind] = 1
            return masks.byte()

        all_mask_ind = get_mask_ind(tgt_tokens, seq_lens)
        itrs = [i for i in range(0, seq_len, step)] if direction == 0 else [i for i in range(seq_len-1, -1, -step)]

        for counter in itrs:
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=self.masking_decision)
            corresponding_probs[pad_mask] = 1.0
            '''
            masks = torch.zeros(*token_probs.shape, device=token_probs.device)
            if direction == 0:
                masks[:, counter:min(counter+step,seq_len)] = 1
            else:
                masks[:, max(counter-step, 0):counter] = 1

            mask_ind = masks.byte() & non_pad_mask
            '''
            mask_ind = select_left(all_mask_ind, counter, step)
            if mask_ind.sum() == 0:
                break

            #print(mask_ind[1].tolist())
            tgt_tokens[mask_ind] = Constants.MASK

            #tqdm.write(("Iteration %d1 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, tags, signal=1)
            # Predict
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            
            #tqdm.write(("Iteration %d2 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())

        for i in range(self.iterations):
            refine_ratio = 0.4 * (1.0 - (i / self.iterations))
            num_mask = (seq_lens.float() * refine_ratio).long()
            mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(
                                                            model, 
                                                            enc_output, 
                                                            category, 
                                                            tgt_tokens, 
                                                            pad_mask, 
                                                            tags)
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())



        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()

        return tgt_tokens, lprobs, (collect_results, collect_scores), None#visual_mask.sum(-1)
    '''

    def get_array_split(self, seq_lens, iterations):
        res = []
        for i in range(seq_lens.size(0)):
            tmp = np.array_split(np.arange(seq_lens[i].cpu()), iterations)
            #print(tmp)
            res.append(tmp)
        return res

    def get_mask_ind_from_array_split(self, tgt_tokens, array_split_info, index):
        masks = torch.zeros(*tgt_tokens.shape, device=tgt_tokens.device)
        for i in range(tgt_tokens.size(0)):
            masks[i, array_split_info[i][index]] = 1
        return masks.byte()

    def generate_sequential(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags, direction, step=1):
        collect_results = []
        collect_scores = []
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        non_pad_mask = tgt_tokens.ne(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        if self.scale == 100:
            tgt_tokens, token_probs, _ = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab, tags)
            #visual_mask = tgt_tokens.ne(Constants.MASK)
            #seq_lens = tgt_tokens.eq(Constants.MASK).sum(dim=1)
        else:
            token_probs = tgt_tokens.new(*tgt_tokens.shape).fill_(0).float()
            token_probs[pad_mask] = 1.0

        array_split_info = self.get_array_split(seq_lens, step)


        if self.collect_best_candidate_iterative_results:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        for counter in range(step):
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=self.masking_decision)
            corresponding_probs[pad_mask] = 1.0
            mask_ind = self.get_mask_ind_from_array_split(tgt_tokens, array_split_info, counter if direction == 0 else (-(counter+1)))
            #print(mask_ind.sum(1))
            #print(mask_ind[1].tolist())
            tgt_tokens[mask_ind] = Constants.MASK

            #tqdm.write(("Iteration %d1 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, tags, signal=1)
            # Predict
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            
            #tqdm.write(("Iteration %d2 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())

        for i in range(self.iterations):
            refine_ratio = 0.4 * (1.0 - (i / self.iterations))
            num_mask = (seq_lens.float() * refine_ratio).long()
            mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(
                                                            model, 
                                                            enc_output, 
                                                            category, 
                                                            tgt_tokens, 
                                                            pad_mask, 
                                                            tags)
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())

        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()

        return tgt_tokens, lprobs, (collect_results, collect_scores), None#visual_mask.sum(-1)
    '''
    
    def generate_easy_first(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags, step=1, refine_ratio=0.2):
        collect_results = []
        collect_scores = []
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        non_pad_mask = tgt_tokens.ne(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        if self.scale == 100:
            tgt_tokens, token_probs, visual_mask = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab, tags)
            #visual_mask = tgt_tokens.ne(Constants.MASK)
            visual_mask = tgt_tokens.ne(Constants.MASK) & tgt_tokens.ne(Constants.PAD)
        else:
            token_probs = tgt_tokens.new(*tgt_tokens.shape).fill_(0).float()
            token_probs[pad_mask] = 1.0
            visual_mask = None

        

        if self.collect_best_candidate_iterative_results:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        iterations = tgt_tokens.eq(Constants.MASK).sum(-1).max() / step
        print(iterations)

        def select_most_confidence(token_probs, mask_ind, step):
            masks = torch.zeros(*token_probs.shape, device=token_probs.device)
            token_probs[~mask_ind] = 0
            remain_length = mask_ind.sum(-1)
            for i in range(masks.size(0)):
                ind = token_probs[i, :].topk(min(step, remain_length[i]), largest=True, sorted=False)[1]
                masks[i, ind] = 1
            return masks.byte()

        counter = 0
        pre = 0
        while True:
            counter += 1
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=self.masking_decision)
            corresponding_probs[pad_mask] = 1.0
            #tqdm.write(("Iteration %dst: " % counter) + ' '.join([('%.3f'%item if item!=1.0 else '') for item in token_probs[1].tolist()]))

            mask_ind = tgt_tokens.eq(Constants.MASK)

            remain = mask_ind.sum()
            if remain == 0 or pre == remain:
                break
            pre = remain

            #tqdm.write(("Iteration %d1 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(
                                                        model, 
                                                        enc_output, 
                                                        category, 
                                                        tgt_tokens, 
                                                        pad_mask, 
                                                        tags)


            most_confidence_ind = select_most_confidence(new_token_probs, mask_ind, step)
            #tqdm.write(("Iteration %dind: " % counter) + ' '.join([('%d'%item) for item in most_confidence_ind[1].tolist()]))
            token_probs[most_confidence_ind] = new_token_probs[most_confidence_ind]
            tgt_tokens[most_confidence_ind] = new_tgt_tokens[most_confidence_ind]
            
            #tqdm.write(("Iteration %d2 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())

        
        for i in range(self.iterations):
            if i == 0 and visual_mask is not None:
                mask_ind = visual_mask
            else:
                num_mask = (seq_lens.float() * refine_ratio).long()
                mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(
                                                            model, 
                                                            enc_output, 
                                                            category, 
                                                            tgt_tokens, 
                                                            pad_mask, 
                                                            tags)
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())
        


        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()

        return tgt_tokens, lprobs, (collect_results, collect_scores), None#visual_mask.sum(-1)
    
    '''
    def easy_first_decode_step(self, tgt_tokens, token_probs, model, enc_output, category, pad_mask, tags, active_inst_idx_list, q):
        def prepare_partial_input(data, active_inst_idx_list):
            assert type(data) == list
            new_data = []
            for item in data:
                if item is None:
                    new_data.append(None)
                else:
                    new_data.append(item.index_select(0, active_inst_idx_list))
            return new_data

        def select_most_confidence(token_probs, mask_ind, q):
            masks = torch.zeros(*token_probs.shape, device=token_probs.device)
            token_probs[~mask_ind] = 0
            remain_length = mask_ind.sum(-1)
            for i in range(masks.size(0)):
                ind = token_probs[i, :].topk(min(q, remain_length[i]), largest=True, sorted=False)[1]
                masks[i, ind] = 1
            return masks.byte()

        def collect_active_inst_idx_list(tgt_tokens, ori_active_inst_idx_list):
            active_inst_idx_list = []
            assert tgt_tokens.size(0) == len(ori_active_inst_idx_list)
            for i in range(tgt_tokens.size(0)):
                is_inst_complete = (tgt_tokens[i].eq(Constants.MASK).gt(0).sum() == 0)
                if not is_inst_complete:
                    active_inst_idx_list.append(ori_active_inst_idx_list[i])
                
            return torch.LongTensor(active_inst_idx_list).to(ori_active_inst_idx_list.device)

        enc_output, category, tgt_tokens, token_probs, pad_mask, tags = prepare_partial_input(
                [enc_output, category, tgt_tokens, token_probs, pad_mask, tags], active_inst_idx_list
            )
        
        new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(
                                                    model, 
                                                    enc_output, 
                                                    category, 
                                                    tgt_tokens, 
                                                    pad_mask, 
                                                    tags)

        # update the most confident q tokens among the unkonwn tokens
        mask_ind = tgt_tokens.eq(Constants.MASK)
        most_confidence_ind = select_most_confidence(new_token_probs, mask_ind, q)
        token_probs[most_confidence_ind] = new_token_probs[most_confidence_ind]
        tgt_tokens[most_confidence_ind] = new_tgt_tokens[most_confidence_ind]
        
        # update the imcompleted instance
        active_inst_idx_list = collect_active_inst_idx_list(tgt_tokens, active_inst_idx_list)
        return active_inst_idx_list, tgt_tokens, token_probs

    def generate_easy_first(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags, step=1, refine_ratio=0.2):
        collect_results = []
        collect_scores = []
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        non_pad_mask = tgt_tokens.ne(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        if self.scale == 100:
            tgt_tokens, token_probs, visual_mask = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab, tags)
            #visual_mask = tgt_tokens.ne(Constants.MASK)
        else:
            token_probs = tgt_tokens.new(*tgt_tokens.shape).fill_(0).float()
            token_probs[pad_mask] = 1.0

        visual_mask = tgt_tokens.ne(Constants.MASK) & tgt_tokens.ne(Constants.PAD)

        if self.collect_best_candidate_iterative_results:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        active_inst_idx_list = torch.LongTensor(list(range(tgt_tokens.size(0)))).to(tgt_tokens.device)
        while True:
            # all instances have finished, i.e., there is no more <mask> token
            if active_inst_idx_list.size(0) == 0:
                break  

            new_active_inst_idx_list, new_tgt_tokens, new_token_probs = self.easy_first_decode_step(
                                                                                tgt_tokens, 
                                                                                token_probs, 
                                                                                model, 
                                                                                enc_output, 
                                                                                category, 
                                                                                pad_mask, 
                                                                                tags,
                                                                                active_inst_idx_list,
                                                                                step
                                                                            )

            # update
            tgt_tokens[active_inst_idx_list] = new_tgt_tokens
            token_probs[active_inst_idx_list] = new_token_probs

            # save results if we need
            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())

            # go to next round until all the instances are done
            active_inst_idx_list = new_active_inst_idx_list

        for i in range(self.iterations):
            if i == 0:
                mask_ind = visual_mask
            else:
                num_mask = (seq_lens.float() * refine_ratio).long()
                mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(
                                                            model, 
                                                            enc_output, 
                                                            category, 
                                                            tgt_tokens, 
                                                            pad_mask, 
                                                            tags)
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())

        

        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()

        return tgt_tokens, lprobs, (collect_results, collect_scores), None#visual_mask.sum(-1)
    
    def generate_easy_first(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags, step=1, refine_ratio=0.2):
        collect_results = []
        collect_scores = []
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        non_pad_mask = tgt_tokens.ne(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        if self.scale == 100:
            tgt_tokens, token_probs, visual_mask = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab, tags)
            #visual_mask = tgt_tokens.ne(Constants.MASK)
            seq_lens = tgt_tokens.eq(Constants.MASK).sum(dim=1)
        else:
            token_probs = tgt_tokens.new(*tgt_tokens.shape).fill_(0).float()
            token_probs[pad_mask] = 1.0

        visual_mask = tgt_tokens.ne(Constants.MASK) & tgt_tokens.ne(Constants.PAD)

        array_split_info = self.get_array_split(seq_lens, step)


        if self.collect_best_candidate_iterative_results:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        def select_most_confidence(token_probs, mask_ind, array_split_info, index):
            masks = torch.zeros(*token_probs.shape, device=token_probs.device)
            token_probs[~mask_ind] = 0
            for i in range(masks.size(0)):
                ind = token_probs[i, :].topk(len(array_split_info[i][index]), largest=True, sorted=False)[1]
                masks[i, ind] = 1
            return masks.byte()

        for counter in range(step):
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=self.masking_decision)
            corresponding_probs[pad_mask] = 1.0
            #tqdm.write(("Iteration %dst: " % counter) + ' '.join([('%.3f'%item if item!=1.0 else '') for item in token_probs[1].tolist()]))

            mask_ind = tgt_tokens.eq(Constants.MASK)
            if mask_ind.sum(-1).gt(0).sum() == 0:
                break

            #tqdm.write(("Iteration %d1 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            new_tgt_tokens, new_token_probs, all_probs = self.generate_non_autoregressive(
                                                        model, 
                                                        enc_output, 
                                                        category, 
                                                        tgt_tokens, 
                                                        pad_mask, 
                                                        tags,
                                                        return_all_probs=True
                                                        )


            most_confidence_ind = select_most_confidence(new_token_probs, mask_ind, array_split_info, counter)
            #tqdm.write(("Iteration %dind: " % counter) + ' '.join([('%d'%item) for item in most_confidence_ind[1].tolist()]))
            token_probs[most_confidence_ind] = new_token_probs[most_confidence_ind]
            tgt_tokens[most_confidence_ind] = new_tgt_tokens[most_confidence_ind]

            #tqdm.write(("Iteration %d2 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())

        
        for i in range(self.iterations):
            if i == 0:
                mask_ind = visual_mask
            else:
                num_mask = (seq_lens.float() * refine_ratio).long()
                mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(
                                                            model, 
                                                            enc_output, 
                                                            category, 
                                                            tgt_tokens, 
                                                            pad_mask, 
                                                            tags)
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())
        


        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()

        return tgt_tokens, lprobs, (collect_results, collect_scores), None#visual_mask.sum(-1)
    '''
    def generate_merge(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags):
        collect_results = []
        collect_scores = []
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        non_pad_mask = tgt_tokens.ne(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        if self.scale == 100:
            tgt_tokens, token_probs, visual_mask = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab, tags)
            #visual_mask = tgt_tokens.ne(Constants.MASK)
        else:
            token_probs = tgt_tokens.new(*tgt_tokens.shape).fill_(0).float()
            token_probs[pad_mask] = 1.0

        if self.collect_best_candidate_iterative_results:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        def select_merge(lens, tgt, idx):
            masks = torch.zeros(*tgt.shape, device=tgt.device)
            left = idx
            right = lens -1 - idx
            for j in range(right.size(0)):
                if left > right[j]:
                    continue
                elif left < right[j]:
                    masks[j, right[j]] = 1
                masks[j, left] = 1
            return masks.byte()

        total_iteration = (seq_len+1)//2
        for i in range(total_iteration):
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=self.masking_decision)
            corresponding_probs[pad_mask] = 1.0
            mask_ind = select_merge(seq_lens, tgt_tokens, i)
            tgt_tokens[mask_ind] = Constants.MASK

            tqdm.write(("Iteration %d1 : " % i) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, tags, signal=1)
            # Predict
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            
            tqdm.write(("Iteration %d2 : " % i) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())


        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()

        return tgt_tokens, lprobs, (collect_results, collect_scores), None#visual_mask.sum(-1)



    def generate_parallel_easy_first(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags):
        collect_results = []
        collect_scores = []
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        non_pad_mask = tgt_tokens.ne(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        tgt_tokens, token_probs, visual_mask = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab, tags)


        iterations = self.iterations if self.scale != 100 else self.iterations + 1

        if self.collect_best_candidate_iterative_results:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        for counter in range(1, iterations):
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=self.masking_decision)
            corresponding_probs[pad_mask] = 1.0
            tqdm.write(("Iteration %dte: " % counter) + ' '.join([('%.3f'%item if item!=1.0 else '') for item in corresponding_probs[1].tolist()]))
            tqdm.write(("Iteration %dst: " % counter) + ' '.join([('%.3f'%item if item!=1.0 else '') for item in token_probs[1].tolist()]))

            '''
            if self.scale == 100:
                if counter == 1:
                    mask_ind = (tgt_tokens == Constants.MASK)
                else:
                    #ratio = max((1.0 - (counter / iterations)), 0.2)
                    ratio = (1.0 - (counter / iterations))
                    #ratio = min((1.0 - (counter / iterations)), 0.7)
                    num_mask = (seq_lens.float() * ratio).long()
                    mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)
            else:
            '''
            mask_ind = self.select_parallel_easy_first(token_probs * corresponding_probs)
            new_input = enlarge(tgt_tokens, seq_len, return_view=False)
            new_input[mask_ind] = Constants.MASK
            new_input = new_input.view(bsz * seq_len, seq_len)

            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(
                                                        model, 
                                                        enlarge(enc_output, seq_len), 
                                                        enlarge(category, seq_len), 
                                                        new_input, 
                                                        enlarge(pad_mask, seq_len), 
                                                        enlarge(tags, seq_len)
                                                        )
            idx = torch.arange(0, seq_len, device=tgt_tokens.device).unsqueeze(0).repeat(bsz, 1).unsqueeze(2)
            new_tgt_tokens = new_tgt_tokens.view(bsz, seq_len, seq_len).gather(2, idx).squeeze(-1)
            new_token_probs = new_token_probs.view(bsz, seq_len, seq_len).gather(2, idx).squeeze(-1)

            tgt_tokens = new_tgt_tokens
            tgt_tokens[pad_mask] = Constants.PAD
            token_probs = new_token_probs
            token_probs[pad_mask] = 1.0
            tqdm.write(("Iteration %d  : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))

            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())

        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()
        
        #lprobs = (token_probs).log()
        return tgt_tokens, lprobs, (collect_results, collect_scores), None#visual_mask.sum(-1)

    def select_parallel_easy_first(self, token_probs):
        # token_probs [batch_size * B, seq_len]
        bsz, seq_len = token_probs.shape
        res, res_idx = token_probs.sort(-1)

        masks = torch.zeros((bsz, seq_len, seq_len), device=res_idx.device)
        for i in range(bsz):
            tmp = res_idx[i]
            for j in range(seq_len):
                masks[i, tmp[j], tmp[:j+1]] = 1

        return masks.byte()

    def generate_mp_refresh(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags):
        collect_results = []
        collect_scores = []
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        iterations = self.iterations if self.scale != 100 else self.iterations + 1
        
        tgt_tokens, token_probs, visual_mask = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab, tags)
        visual_probs = token_probs[visual_mask]

        tmp_token_probs = token_probs.clone()

        if self.collect_best_candidate_iterative_results and not self.collect_last:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        for counter in range(1, iterations):
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=self.masking_decision)
            corresponding_probs[pad_mask] = 1.0
            #tqdm.write(("Iteration %dte: " % counter) + ' '.join([('%.3f'%item if item!=1.0 else '') for item in corresponding_probs[1].tolist()]))
            #tqdm.write(("Iteration %dst: " % counter) + ' '.join([('%.3f'%item if item!=1.0 else '') for item in token_probs[1].tolist()]))

            if self.scale == 100:
                if counter == 1:
                    mask_ind = (tgt_tokens == Constants.MASK)
                else:
                    #ratio = max((1.0 - (counter / iterations)), 0.3)
                    ratio = (1.0 - (counter / iterations))
                    #ratio = 0.4
                    #ratio = min((1.0 - (counter / iterations)), 0.7)
                    num_mask = (seq_lens.float() * ratio).long()
                    mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)
                    #mask_ind = self.select_worst(token_probs, num_mask)
            else:
                #ratio = max((1.0 - (counter / iterations)), 0.2)
                ratio = (1.0 - (counter / iterations))
                #ratio = min((1.0 - (counter / iterations)), 0.7)
                num_mask = (seq_lens.float() * ratio).long()
                mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)
                #mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            #tqdm.write(("Iteration %d1 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            # Predict
            new_tgt_tokens, new_token_probs, all_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, tags, signal=1, return_all_probs=True)
            #print(all_probs.shape, tgt_tokens.shape)
            #non_mask_ind = ~mask_ind
            #token_probs[non_mask_ind] = all_probs.gather(2, tgt_tokens[non_mask_ind].unsqueeze(2)).squeeze(2)

            #tqdm.write(("Iteration %d0 : " % counter) + to_sentence(new_tgt_tokens[1].tolist(), tgt_vocab))
            # Predict
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tmp_token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]

            token_probs[mask_ind] = 0
            token_probs = torch.max(token_probs, all_probs.gather(2, tgt_tokens.unsqueeze(2)).squeeze(2))
            #token_probs = (token_probs + all_probs.gather(2, tgt_tokens.unsqueeze(2)).squeeze(2))/2
            token_probs[pad_mask] = 1.0



            #tqdm.write(("Iteration %d2 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            if self.collect_best_candidate_iterative_results and not self.collect_last:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())


        if self.collect_last:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        #token_probs = tmp_token_probs
        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()
        
        #lprobs = (token_probs).log()
        return tgt_tokens, lprobs, (collect_results, collect_scores), None#visual_mask.sum(-1)

    def generate_fix_tokens(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags):
        collect_results = []
        collect_scores = []
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        iterations = self.iterations if self.scale != 100 else self.iterations + 1
        
        tgt_tokens, token_probs, visual_mask = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab, tags)
        visual_probs = token_probs[visual_mask]

        tmp_token_probs = token_probs.clone()
        all_ratio = [0.666666666666666, 0.5, 0.3, 0.3] + [0.3] * 10
        #all_ratio = [0.1] * 10
        index = 0
        if self.collect_best_candidate_iterative_results and not self.collect_last:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        for counter in range(1, iterations):
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=self.masking_decision)
            corresponding_probs[pad_mask] = 1.0
            #tqdm.write(("Iteration %dte: " % counter) + ' '.join([('%.3f'%item if item!=1.0 else '') for item in corresponding_probs[1].tolist()]))
            #tqdm.write(("Iteration %dst: " % counter) + ' '.join([('%.3f'%item if item!=1.0 else '') for item in token_probs[1].tolist()]))

            if self.scale == 100:
                if counter == 1:
                    mask_ind = (tgt_tokens == Constants.MASK)
                else:
                    #ratio = max((1.0 - (counter / iterations)), 0.3)
                    #ratio = (1.0 - (counter / iterations))
                    ratio = all_ratio[index]
                    index += 1
                    #tqdm.write("%s"%str(ratio))
                    #ratio = 0.4
                    #ratio = min((1.0 - (counter / iterations)), 0.7)
                    num_mask = (seq_lens.float() * ratio).long()
                    #num_mask[num_mask < 1] = 1
                    #mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask, limit=0.35, seq_lens=seq_lens)
                    mask_ind = self.select_worst(token_probs, num_mask)
            else:
                #ratio = max((1.0 - (counter / iterations)), 0.2)
                ratio = 0.3
                #ratio = min((1.0 - (counter / iterations)), 0.7)
                num_mask = (seq_lens.float() * ratio).long()
                mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)
                #mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            #tqdm.write(("Iteration %d1 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            # Predict
            new_tgt_tokens, new_token_probs, all_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, tags, signal=1, return_all_probs=True)
            #print(all_probs.shape, tgt_tokens.shape)
            #non_mask_ind = ~mask_ind
            #token_probs[non_mask_ind] = all_probs.gather(2, tgt_tokens[non_mask_ind].unsqueeze(2)).squeeze(2)

            #tqdm.write(("Iteration %d0 : " % counter) + to_sentence(new_tgt_tokens[1].tolist(), tgt_vocab))
            # Predict
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]



            #tqdm.write(("Iteration %d2 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            if self.collect_best_candidate_iterative_results and not self.collect_last:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())


        if self.collect_last:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        #token_probs = tmp_token_probs
        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()
        
        #lprobs = (token_probs).log()
        return tgt_tokens, lprobs, (collect_results, collect_scores), None#visual_mask.sum(-1)



    def generate(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags):
        if self.paradigm == 'mp':
            return self.generate_mp(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags)
        elif self.paradigm == 'ap':
            return self.generate_ap(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags)
        elif self.paradigm == 'l2r':
            return self.generate_sequential(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags, direction=0, step=self.q)
        elif self.paradigm == 'r2l':
            return self.generate_sequential(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags, direction=1)
        elif self.paradigm == 'ef':
            return self.generate_easy_first(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags, step=self.q)
        elif self.paradigm == 'pef':
            return self.generate_parallel_easy_first(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags)
        elif self.paradigm == 'merge':
            return self.generate_merge(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags)
        elif self.paradigm == 'mpr':
            return self.generate_mp_refresh(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags)
        elif self.paradigm == 'ft':
            return self.generate_fix_tokens(model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab, tags)
    
    def generate_non_autoregressive(self, model, enc_output, category, tgt_tokens, pad_mask, tags, signal=0, zeros=[], tag_replace=None, return_all_probs=False):
        decoder_out, _ = model.decoder.forward_(tgt_tokens, enc_output, category, signal=signal, tags=tags)
        tgt_tokens, token_probs, all_probs = generate_step_with_prob(model.tgt_word_prj(decoder_out), zeros=zeros)

        tgt_tokens[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0

        if return_all_probs:
            return tgt_tokens, token_probs, all_probs

        if tag_replace is not None:
            source, target = tag_replace
            ind = tgt_tokens.eq(source)
            tgt_tokens[ind] = target
            copy_ = token_probs.clone()
            token_probs[ind] = 0.0
            return tgt_tokens, token_probs, copy_
        return tgt_tokens, token_probs

    def mapping(self, tgt_tokens):
        tokens = tgt_tokens.clone().flatten()
        for i, token in enumerate(tokens):
            tokens[i] = self.dict_mapping[token.item()]
        return tokens.view(*tgt_tokens.shape)

    def scoring_by_teacher(self, teacher_model, teacher_enc_output, category, tgt_tokens, decision=True):
        if teacher_model is None or not decision:
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



    def select_worst(self, token_probs, num_mask, limit=None, seq_lens=None):
        masks = torch.zeros(*token_probs.shape, device=token_probs.device)
        if limit is not None:
            #tmp = []
            token_probs = token_probs.log()
            for i in range(masks.size(0)):
                #tmp = (token_probs[i, :] < limit)
                tmp = token_probs[i, :].sum() / seq_lens[i]
                tmp = token_probs[i, :] < tmp
                
                if tmp.sum() < num_mask[i] and tmp.sum() != 0:
                    masks[i] = tmp
                else:
                    ind = token_probs[i, :].topk(max(1, num_mask[i]), largest=False, sorted=False)[1]
                    masks[i, ind] = 1
                
            #print(max(tmp), min(tmp), sum(tmp)/len(tmp))
        else:
            #tmp = []
            for i in range(masks.size(0)):
                #tmp.append(token_probs[i, :].topk(max(1, num_mask[i]), largest=False, sorted=False)[0][-1].data)
                ind = token_probs[i, :].topk(max(1, num_mask[i]), largest=False, sorted=False)[1]
                masks[i, ind] = 1
            #print(max(tmp), min(tmp), sum(tmp)/len(tmp))
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
'''

class NV(object):
    
    def __init__(self, iterations, seed, dict_mapping, plot=False, collect_best_candidate_iterative_results=False, **kwargs):
        super().__init__()
        self.iterations = iterations
        self.random = np.random.RandomState(seed)
        self.dict_mapping = dict_mapping
        self.plot = plot
        self.collect_best_candidate_iterative_results = collect_best_candidate_iterative_results
        opt = kwargs['opt']
        self.visual_tag = opt['visual_tag']
        self.nonvisual_tag = opt['nonvisual_tag']
        self.revision_tag = opt['revision_tag']


    def separation_integration(self, model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab):
        mask_ind = tgt_tokens.eq(Constants.MASK)
        t1, t2 = tgt_tokens.clone(), tgt_tokens
        t1[mask_ind] = self.visual_tag

        t1, t1_probs = self.generate_non_autoregressive(model, enc_output, category, t1, pad_mask, signal=0)
        tqdm.write("    Visual   : " + to_sentence(t1[1].tolist(), tgt_vocab))
        t2, t2_probs = self.generate_non_autoregressive(model, enc_output, category, t2, pad_mask, signal=1)
        tqdm.write("    Mask     : " + to_sentence(t2[1].tolist(), tgt_vocab))

        t1_probs[t1.eq(Constants.MASK)] = 0.0
        ind = t1_probs > t2_probs
        t2[ind] = t1[ind]
        t2_probs[ind] = t1_probs[ind]


        tqdm.write("    Fusion   : " + to_sentence(t2[1].tolist(), tgt_vocab))

        return t2, t2_probs

    def generate(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        collect_results = []
        
        iterations = self.iterations
        
        tgt_tokens, token_probs = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab)

        for counter in range(1, iterations):
            #corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens)
            #corresponding_probs[pad_mask] = 1.0

            ratio = (1.0 - (counter / iterations))
            num_mask = (seq_lens.float() * ratio).long()
            #mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)
            mask_ind = self.select_worst(token_probs, num_mask)
            tgt_tokens[mask_ind] = Constants.MASK

            # Predict
            tgt_tokens, token_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, signal=1)

            tqdm.write(("Iteration %d: " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))

        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens)
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()
        
        #lprobs = (token_probs).log()
        return tgt_tokens, lprobs, collect_results
    
    def generate_non_autoregressive(self, model, enc_output, category, tgt_tokens, pad_mask, signal, zeros=[], tag_replace=None):
        decoder_out = model.decoder.forward_(tgt_tokens, enc_output, category, signal=signal)
        tgt_tokens, token_probs, all_probs = generate_step_with_prob(model.tgt_word_prj(decoder_out), zeros=zeros)

        tgt_tokens[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0

        if tag_replace is not None:
            source, target = tag_replace
            ind = tgt_tokens.eq(source)
            tgt_tokens[ind] = target
            copy_ = token_probs.clone()
            token_probs[ind] = 0.0
            return tgt_tokens, token_probs, copy_
        return tgt_tokens, token_probs

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
'''
class MS(object):
    
    def __init__(self, iterations, seed, dict_mapping, plot=False, collect_best_candidate_iterative_results=False, **kwargs):
        super().__init__()
        self.iterations = iterations
        self.random = np.random.RandomState(seed)
        self.dict_mapping = dict_mapping
        self.plot = plot
        self.collect_best_candidate_iterative_results = collect_best_candidate_iterative_results
        opt = kwargs['opt']
        self.visual_tag = opt['visual_tag']
        self.nonvisual_tag = opt['nonvisual_tag']
        self.revision_tag = opt['revision_tag']
        self.masking_decision = opt.get('masking_decision', False)
        self.no_candidate_decision = opt.get('no_candidate_decision', False)
        self.collect_best_candidate_iterative_results = opt.get('collect_best_candidate_iterative_results', False)

        self.scale = opt.get('nv_scale', 0.0)
        self.fixed_iterations = opt.get('fixed_iterations', -1)
        self.multiscale = opt['multiscale']

        if self.fixed_iterations != -1: assert self.scale > 0
        assert self.fixed_iterations <= self.iterations - 2


    def separation_integration(self, model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab):
        mask_ind = tgt_tokens.eq(Constants.MASK)
        t1, t2 = tgt_tokens.clone(), tgt_tokens
        t1[mask_ind] = self.visual_tag

        t1, t1_probs = self.generate_non_autoregressive(model, enc_output, category, t1, enlarge(pad_mask, self.multiscale), multiscale=True)

        tmp = t1.view(-1, self.multiscale, t1.size(-1))
        res = tmp.chunk(self.multiscale, dim=1)
        for i in range(len(res)):
            tqdm.write(("    Visual%d   : " % i) + to_sentence(res[i][1][0].tolist(), tgt_vocab))
        
        t2, t2_probs = self.generate_non_autoregressive(model, enc_output, category, t2, pad_mask, multiscale=False)
        tqdm.write("    Mask     : " + to_sentence(t2[1].tolist(), tgt_vocab))


        tqdm.write("    Fusion   : " + to_sentence(t2[1].tolist(), tgt_vocab))

        return t2, t2_probs, None
        #return t1, t1_probs

    def generate(self, model, teacher_model, enc_output, teacher_enc_output, category, tgt_tokens, tgt_vocab):
        collect_results = []
        collect_scores = []
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens1 = seq_len - pad_mask.sum(dim=1)
        
        iterations = self.iterations
        
        tgt_tokens, token_probs, visual_mask = self.separation_integration(model, enc_output, category, tgt_tokens, pad_mask, tgt_vocab)
        visual_probs = token_probs[visual_mask]
        seq_lens2 = seq_lens1 - visual_mask.sum(-1) if visual_mask is not None else seq_lens1
        #if visual_mask is not None:
        #    seq_lens = seq_lens - visual_mask.sum(-1)
        #print(visual_mask.long().sum(-1).float())

        if self.collect_best_candidate_iterative_results:
            collect_results.append(tgt_tokens.clone())
            collect_scores.append(token_probs.clone())

        for counter in range(1, iterations):
            corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=self.masking_decision)
            corresponding_probs[pad_mask] = 1.0

            seq_lens = seq_lens1

            #ratio = max((1.0 - (counter / iterations)), 0.2)
            ratio = (1.0 - (counter / iterations))
            #ratio = min((1.0 - (counter / iterations)), 0.7)
            num_mask = (seq_lens.float() * ratio).long()
            mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)
            #mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            tqdm.write(("Iteration %d1 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            # Predict
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(model, enc_output, category, tgt_tokens, pad_mask, multiscale=False)

            # Predict
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            tqdm.write(("Iteration %d2 : " % counter) + to_sentence(tgt_tokens[1].tolist(), tgt_vocab))
            if self.collect_best_candidate_iterative_results:
                collect_results.append(tgt_tokens.clone())
                collect_scores.append(token_probs.clone())

        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_enc_output, category, tgt_tokens, decision=(not self.no_candidate_decision))
        corresponding_probs[pad_mask] = 1.0
        lprobs = (token_probs * corresponding_probs).log()
        
        #lprobs = (token_probs).log()
        return tgt_tokens, lprobs, (collect_results, collect_scores), None#visual_mask.sum(-1)
    
    def generate_non_autoregressive(self, model, enc_output, category, tgt_tokens, pad_mask, multiscale, zeros=[], tag_replace=None):
        decoder_out = model.decoder.forward_(tgt_tokens, enc_output, category, multiscale=multiscale)
        tgt_tokens, token_probs, all_probs = generate_step_with_prob(model.tgt_word_prj(decoder_out), zeros=zeros)

        tgt_tokens[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0

        if tag_replace is not None:
            source, target = tag_replace
            ind = tgt_tokens.eq(source)
            tgt_tokens[ind] = target
            copy_ = token_probs.clone()
            token_probs[ind] = 0.0
            return tgt_tokens, token_probs, copy_
        return tgt_tokens, token_probs

    def mapping(self, tgt_tokens):
        tokens = tgt_tokens.clone().flatten()
        for i, token in enumerate(tokens):
            tokens[i] = self.dict_mapping[token.item()]
        return tokens.view(*tgt_tokens.shape)

    def scoring_by_teacher(self, teacher_model, teacher_enc_output, category, tgt_tokens, decision=True):
        if teacher_model is None or not decision:
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