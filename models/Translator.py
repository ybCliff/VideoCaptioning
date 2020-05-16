''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.Beam import Beam
import os, json
import models.Constants as Constants
class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, model, opt, device=torch.device('cuda'), teacher_model=None, dict_mapping={}):
        self.model = model
        self.model.eval()
        self.opt = opt
        self.device = device
        self.teacher_model = teacher_model
        self.dict_mapping = dict_mapping
        self.length_bias = opt.get('length_bias', 0)


    def get_inst_idx_to_tensor_position_map(self, inst_idx_list):
        ''' Indicate the position of an instance in a tensor. '''
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(self, beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        ''' Collect tensor parts associated to active instances. '''

        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        #print('n_prev_active:', n_prev_active_inst)
        #print('n_curr_active:', curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(self, enc_output, inst_idx_to_position_map, active_inst_idx_list, category, n_bm, enc_hidden=None, tag=None):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

        if isinstance(enc_output, list):
            active_src_enc = []
            for item in enc_output:
                active_src_enc.append(self.collect_active_part(item, active_inst_idx, n_prev_active_inst, n_bm))
        else:
            active_src_enc = self.collect_active_part(enc_output, active_inst_idx, n_prev_active_inst, n_bm)
        active_category = self.collect_active_part(category, active_inst_idx, n_prev_active_inst, n_bm)

        if enc_hidden is not None:
            if isinstance(enc_hidden, list):
                active_hidden = []
                for i in range(len(enc_hidden)):
                    assert isinstance(enc_hidden[i], tuple)
                    tmp1 = self.collect_active_part(enc_hidden[i][0], active_inst_idx, n_prev_active_inst, n_bm)
                    tmp2 = self.collect_active_part(enc_hidden[i][1], active_inst_idx, n_prev_active_inst, n_bm)
                    active_hidden.append((tmp1, tmp2))
            else:
                assert isinstance(enc_hidden, tuple)
                tmp1 = self.collect_active_part(enc_hidden[0], active_inst_idx, n_prev_active_inst, n_bm)
                tmp2 = self.collect_active_part(enc_hidden[1], active_inst_idx, n_prev_active_inst, n_bm)
                active_hidden = (tmp1, tmp2)

        active_tag = None
        if tag is not None:
            active_tag = self.collect_active_part(tag, active_inst_idx, n_prev_active_inst, n_bm)


        active_inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        if enc_hidden is None:
            #if tag is not None:
            return active_src_enc, active_category, active_inst_idx_to_position_map, active_tag
            #return active_src_enc, active_category, active_inst_idx_to_position_map
        
        return active_src_enc, active_hidden, active_category, active_inst_idx_to_position_map, active_tag


    def collect_active_inst_idx_list(self, inst_beams, word_prob, inst_idx_to_position_map):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])

            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]
        return active_inst_idx_list

    def collect_hypothesis_and_scores(self, inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tk = inst_dec_beams[inst_idx].sort_finished(self.opt.get('beam_alpha', 1.0))
            n_best = min(n_best, len(scores))
            all_scores += [scores[:n_best]]
            hyps = [inst_dec_beams[inst_idx].get_hypothesis_from_tk(t, k) for t, k in tk[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    def translate_batch_ARFormer(self, encoder_outputs, category):
        ''' Translation work in one batch '''

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm, category, attribute):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                #print(dec_partial_seq)
                return dec_partial_seq

            def predict_word(dec_seq, enc_output, n_active_inst, n_bm, category, attribute):
                dec_output, *_ = self.model.decoder(dec_seq, enc_output, category, tags=attribute)
                if isinstance(dec_output, list):
                    dec_output = dec_output[-1]
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(self.model.tgt_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

                return word_prob

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            word_prob = predict_word(dec_seq, enc_output, n_active_inst, n_bm, category, attribute)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = self.collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        '''
        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                print(113, scores, tail_idxs)
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores
        '''

        with torch.no_grad():
            enc_output = encoder_outputs['enc_output']
            if isinstance(enc_output, list):
                assert len(enc_output) == 1
                enc_output = enc_output[0]
            #-- Repeat data for beam search
            n_bm = self.opt["beam_size"]
            n_inst, len_s, d_h = enc_output.size()
            enc_output = enc_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
            category = category.repeat(1, n_bm).view(n_inst * n_bm, 1)

            attribute = encoder_outputs.get(Constants.mapping['attr'][0], None)
            if attribute is not None: attribute = attribute.unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, -1)

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, self.opt["max_len"], device=self.device, specific_nums_of_sents=self.opt.get('topk', 1)) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, self.opt["max_len"]):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm, category, attribute)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                enc_output, category, inst_idx_to_position_map, attribute = self.collate_active_info(
                    enc_output, inst_idx_to_position_map, active_inst_idx_list, category, n_bm, tag=attribute)

        batch_hyp, batch_scores = self.collect_hypothesis_and_scores(inst_dec_beams, self.opt.get("topk", 1))

        return batch_hyp, batch_scores

    def translate_batch_LSTM(self, encoder_outputs, category):
        ''' Translation work in one batch '''
        def beam_decode_step(
                inst_dec_beams, enc_output, enc_hidden, inst_idx_to_position_map, n_bm, category, tag):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams):
                dec_partial_seq = [b.get_lastest_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1)
                #print(dec_partial_seq)
                return dec_partial_seq

            def predict_word(dec_seq, enc_output, enc_hidden, n_active_inst, n_bm, category, tag):
                res = self.model.decoder(
                            it=dec_seq, 
                            encoder_outputs=enc_output, 
                            category=category, 
                            decoder_hidden=enc_hidden, 
                            tag=tag
                        )
                dec_output, enc_hidden, tag = res['dec_outputs'], res['dec_hidden'], res['pred_tag']
                word_prob = F.log_softmax(self.model.tgt_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                return word_prob, enc_hidden, tag.argmax(1) if tag is not None else None


            def collect_active_hidden_single(inst_beams, inst_idx_to_position_map, enc_hidden, n_bm):
                if isinstance(enc_hidden, tuple):
                    tmp1, tmp2 = enc_hidden
                    _, *d_hs = tmp1.size()
                    n_curr_active_inst = len(inst_idx_to_position_map)
                    new_shape = (n_curr_active_inst * n_bm, *d_hs)
                    tmp1 = tmp1.view(n_curr_active_inst, n_bm, -1)
                    tmp2 = tmp2.view(n_curr_active_inst, n_bm, -1)
                    #print('hidden:', tmp1)

                    for inst_idx, inst_position in inst_idx_to_position_map.items():
                        _prev_ks = inst_beams[inst_idx].get_current_origin()
                        tmp1[inst_position] = tmp1[inst_position].index_select(0, _prev_ks)
                        tmp2[inst_position] = tmp2[inst_position].index_select(0, _prev_ks)
                        #print("PREV_KS:", _prev_ks)
                    #print('after h:', tmp1)
                    tmp1 = tmp1.view(*new_shape)
                    tmp2 = tmp2.view(*new_shape)
                    enc_hidden = (tmp1, tmp2)
                else:
                    _, *d_hs = enc_hidden.size()
                    n_curr_active_inst = len(inst_idx_to_position_map)
                    new_shape = (n_curr_active_inst * n_bm, *d_hs)
                    enc_hidden = enc_hidden.view(n_curr_active_inst, n_bm, -1)

                    for inst_idx, inst_position in inst_idx_to_position_map.items():
                        _prev_ks = inst_beams[inst_idx].get_current_origin()
                        enc_hidden[inst_position] = enc_hidden[inst_position].index_select(0, _prev_ks)

                    enc_hidden = enc_hidden.view(*new_shape)
 
                return enc_hidden 

            def collect_active_hidden(inst_beams, inst_idx_to_position_map, enc_hidden, n_bm):
                if enc_hidden is None:
                    return None
                if isinstance(enc_hidden, list):
                    hidden = []
                    for item in enc_hidden:
                        hidden.append(collect_active_hidden_single(inst_beams, inst_idx_to_position_map, item, n_bm))
                else:
                    hidden = collect_active_hidden_single(inst_beams, inst_idx_to_position_map, enc_hidden, n_bm)
                return hidden
                
                '''
                _, *d_hs = beamed_tensor.size()
                n_curr_active_inst = len(curr_active_inst_idx)
                new_shape = (n_curr_active_inst * n_bm, *d_hs)

                print('n_prev_active:', n_prev_active_inst)
                print('n_curr_active:', curr_active_inst_idx)
                beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
                beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
                beamed_tensor = beamed_tensor.view(*new_shape)

                return beamed_tensor
                '''


            n_active_inst = len(inst_idx_to_position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams)
            #print(dec_seq)
            #print('before:', enc_hidden[0])
            word_prob, enc_hidden, tag = predict_word(dec_seq, enc_output, enc_hidden, n_active_inst, n_bm, category, tag)
            #print('after:', enc_hidden[0])
            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = self.collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            enc_hidden = collect_active_hidden(inst_dec_beams, inst_idx_to_position_map, enc_hidden, n_bm)
            tag = collect_active_hidden(inst_dec_beams, inst_idx_to_position_map, tag, n_bm)
            return active_inst_idx_list, enc_hidden, tag

        with torch.no_grad():
            enc_output, enc_hidden = encoder_outputs['enc_output'], encoder_outputs['enc_hidden']
            if not isinstance(enc_output, list):
                enc_output = [enc_output]

            n_bm = self.opt["beam_size"]
            n_inst, len_s, _ = enc_output[0].shape

            
            #-- Repeat data for beam search
            enc_output = [item.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, -1) for item in enc_output]
            if isinstance(enc_hidden, tuple):
                n_inst, d_h = enc_hidden[0].size()
                enc_hidden = (enc_hidden[0].unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, d_h), enc_hidden[1].unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, d_h))
            elif isinstance(enc_hidden, list):
                n_inst, d_h = enc_hidden[0].size()
                enc_hidden = [item.unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, d_h) for item in enc_hidden]
            else:
                n_inst, d_h = enc_hidden.size()
                enc_hidden = enc_hidden.unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, d_h)
            enc_hidden = self.model.decoder.init_hidden(enc_hidden)
            if encoder_outputs.get('obj_emb', None) is not None:
                if self.opt['with_category']:
                    category = torch.cat([category, encoder_outputs['obj_emb']], dim=1)
                else:
                    category = encoder_outputs['obj_emb']

            category = category.unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, -1)

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, self.opt["max_len"], device=self.device) for _ in range(n_inst)]
            
            if self.opt['use_tag']:
                tag = category.new(n_inst, n_bm).fill_(Constants.BOS).view(n_inst * n_bm).long()
            else:
                tag = None

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for t in range(1, self.opt["max_len"]):
                active_inst_idx_list, enc_hidden, tag = beam_decode_step(
                    inst_dec_beams, enc_output, enc_hidden, inst_idx_to_position_map, n_bm, category, tag)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                enc_output, enc_hidden, category, inst_idx_to_position_map, tag = self.collate_active_info(
                    enc_output, inst_idx_to_position_map, active_inst_idx_list, category, n_bm, enc_hidden=enc_hidden, tag=tag)

        batch_hyp, batch_scores = self.collect_hypothesis_and_scores(inst_dec_beams, self.opt.get("topk", 1))

        return batch_hyp, batch_scores

    def translate_batch_ENSEMBLE(self, encoder_outputs, category):
        ''' Translation work in one batch '''
        def beam_decode_step(
                inst_dec_beams, enc_output, enc_hidden, inst_idx_to_position_map, n_bm, category):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams):
                dec_partial_seq = [b.get_lastest_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1)
                #print(dec_partial_seq)
                return dec_partial_seq

            def predict_word(dec_seq, enc_output, enc_hidden, n_active_inst, n_bm, category):
                dec_output, enc_hidden, *_ = self.model.decoder(dec_seq, enc_output, category, enc_hidden)
                assert isinstance(dec_output, list)
                word_prob = []
                for i in range(len(dec_output)):
                    tmp = F.log_softmax(self.model.tgt_word_prj(dec_output[i]), dim=1)
                    tmp = tmp.view(n_active_inst, n_bm, -1)
                    word_prob.append(tmp)
                word_prob = torch.stack(word_prob, dim=0).mean(0)

                return word_prob, enc_hidden

            def collect_active_hidden(inst_beams, inst_idx_to_position_map, enc_hidden, n_bm):
                assert isinstance(enc_hidden, list)
                n_curr_active_inst = len(inst_idx_to_position_map)

                for i in range(len(enc_hidden)):
                    tmp1, tmp2 = enc_hidden[i]
                    _, *d_hs = tmp1.size()
                    
                    new_shape = (n_curr_active_inst * n_bm, *d_hs)
                    tmp1 = tmp1.view(n_curr_active_inst, n_bm, -1)
                    tmp2 = tmp2.view(n_curr_active_inst, n_bm, -1)
                    #print('hidden:', tmp1)

                    for inst_idx, inst_position in inst_idx_to_position_map.items():
                        _prev_ks = inst_beams[inst_idx].get_current_origin()
                        tmp1[inst_position] = tmp1[inst_position].index_select(0, _prev_ks)
                        tmp2[inst_position] = tmp2[inst_position].index_select(0, _prev_ks)
                        #print("PREV_KS:", _prev_ks)
                    #print('after h:', tmp1)
                    tmp1 = tmp1.view(*new_shape)
                    tmp2 = tmp2.view(*new_shape)
                    enc_hidden[i] = (tmp1, tmp2)
                return enc_hidden 


            n_active_inst = len(inst_idx_to_position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams)
            #print(dec_seq)
            #print('before:', enc_hidden[0])
            word_prob, enc_hidden = predict_word(dec_seq, enc_output, enc_hidden, n_active_inst, n_bm, category)
            #print('after:', enc_hidden[0])
            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = self.collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            enc_hidden = collect_active_hidden(inst_dec_beams, inst_idx_to_position_map, enc_hidden, n_bm)
            return active_inst_idx_list, enc_hidden

        with torch.no_grad():
            enc_output, enc_hidden = encoder_outputs['enc_output'], encoder_outputs['enc_hidden']
            if not isinstance(enc_output, list):
                enc_output = [enc_output]

            n_bm = self.opt["beam_size"]
            n_inst, len_s, d_h = enc_output[0].size()

            
            #-- Repeat data for beam search
            
            enc_output = [item.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h) for item in enc_output]

            assert isinstance(enc_hidden, list)
            for i in range(len(enc_hidden)):
                if isinstance(enc_hidden[i], tuple):
                    enc_hidden[i] = (enc_hidden[i][0].unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, d_h), 
                        enc_hidden[i][1].unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, d_h))
                else:
                    enc_hidden[i] = enc_hidden[i].unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, d_h)
            enc_hidden = self.model.decoder.init_hidden(enc_hidden)
            category = category.unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, self.opt['num_category'])

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, self.opt["max_len"], device=self.device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for t in range(1, self.opt["max_len"]):
                active_inst_idx_list, enc_hidden = beam_decode_step(
                    inst_dec_beams, enc_output, enc_hidden, inst_idx_to_position_map, n_bm, category)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                enc_output, enc_hidden, category, inst_idx_to_position_map = self.collate_active_info(
                    enc_output, inst_idx_to_position_map, active_inst_idx_list, category, n_bm, enc_hidden=enc_hidden)

        batch_hyp, batch_scores = self.collect_hypothesis_and_scores(inst_dec_beams, self.opt.get("topk", 1))

        return batch_hyp, batch_scores

    def translate_batch_NARFormer(self, encoder_outputs, category, tgt_tokens, tgt_vocab, teacher_encoder_outputs, tags):
        from decoding.mask_predict import generate

        with torch.no_grad():
            return generate(
                        model=self.model,
                        teacher_model=self.teacher_model,
                        encoder_outputs=encoder_outputs, 
                        teacher_encoder_outputs=teacher_encoder_outputs,
                        category=category, 
                        tgt_tokens=tgt_tokens, 
                        tgt_vocab=tgt_vocab, 
                        opt=self.opt,
                        dict_mapping=self.dict_mapping,
                        length_bias=self.length_bias,
                        tags=tags#encoder_outputs.get(Constants.mapping['attr'][0], None)#tags
                    )



    def translate_batch(self, encoder_outputs, category, tgt_tokens, tgt_vocab, teacher_encoder_outputs=None, tags=None):
        if self.opt['decoder_type'] == 'NARFormer':
            return self.translate_batch_NARFormer(encoder_outputs, category, tgt_tokens, tgt_vocab, teacher_encoder_outputs, tags=tags)

        func_mapping = {
            'LSTM': self.translate_batch_LSTM,
            'ARFormer': self.translate_batch_ARFormer,
            'ENSEMBLE': self.translate_batch_ENSEMBLE
        }
        return func_mapping[self.opt['decoder_type']](encoder_outputs, category)

class Translator_ensemble(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, model, opt, device=torch.device('cuda')):
        self.model = model
        assert isinstance(model, list)
        for m in self.model:
            m.eval()   
        self.opt = opt
        self.device = device


    def get_inst_idx_to_tensor_position_map(self, inst_idx_list):
        ''' Indicate the position of an instance in a tensor. '''
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(self, beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        ''' Collect tensor parts associated to active instances. '''

        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        #print('n_prev_active:', n_prev_active_inst)
        #print('n_curr_active:', curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(self, enc_output, inst_idx_to_position_map, active_inst_idx_list, category, n_bm, enc_hidden=None):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

        if isinstance(enc_output, list):
            active_src_enc = []
            for item in enc_output:
                tmp = []
                for i in item:
                    tmp.append(self.collect_active_part(i, active_inst_idx, n_prev_active_inst, n_bm))
                active_src_enc.append(tmp)
        else:
            active_src_enc = self.collect_active_part(enc_output, active_inst_idx, n_prev_active_inst, n_bm)
        active_category = self.collect_active_part(category, active_inst_idx, n_prev_active_inst, n_bm)

        if enc_hidden is not None:
            if isinstance(enc_hidden, list):
                active_hidden = []
                for i in range(len(enc_hidden)):
                    assert isinstance(enc_hidden[i], tuple)
                    tmp1 = self.collect_active_part(enc_hidden[i][0], active_inst_idx, n_prev_active_inst, n_bm)
                    tmp2 = self.collect_active_part(enc_hidden[i][1], active_inst_idx, n_prev_active_inst, n_bm)
                    active_hidden.append((tmp1, tmp2))
            else:
                assert isinstance(enc_hidden, tuple)
                tmp1 = self.collect_active_part(enc_hidden[0], active_inst_idx, n_prev_active_inst, n_bm)
                tmp2 = self.collect_active_part(enc_hidden[1], active_inst_idx, n_prev_active_inst, n_bm)
                active_hidden = (tmp1, tmp2)

        active_inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        if enc_hidden is None:
            return active_src_enc, active_category, active_inst_idx_to_position_map
        return active_src_enc, active_hidden, active_category, active_inst_idx_to_position_map

    def collect_active_inst_idx_list(self, inst_beams, word_prob, inst_idx_to_position_map):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])

            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]
        return active_inst_idx_list

    def collect_hypothesis_and_scores(self, inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tk = inst_dec_beams[inst_idx].sort_finished(self.opt.get('beam_alpha', 1.0))
            n_best = min(n_best, len(scores))
            all_scores += [scores[:n_best]]
            hyps = [inst_dec_beams[inst_idx].get_hypothesis_from_tk(t, k) for t, k in tk[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    def translate_batch_ENSEMBLE(self, enc_output, enc_hidden, category):
        ''' Translation work in one batch '''
        def beam_decode_step(
                inst_dec_beams, enc_output, enc_hidden, inst_idx_to_position_map, n_bm, category):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams):
                dec_partial_seq = [b.get_lastest_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1)
                #print(dec_partial_seq)
                return dec_partial_seq

            def predict_word(dec_seq, enc_output, enc_hidden, n_active_inst, n_bm, category):
                word_prob = []
                for i in range(len(enc_output)):
                    dec_output, enc_hidden[i], *_ = self.model[i].decoder(dec_seq, enc_output[i], category, enc_hidden[i])
                    tmp = F.log_softmax(self.model[i].tgt_word_prj(dec_output), dim=1)
                    tmp = tmp.view(n_active_inst, n_bm, -1)
                    word_prob.append(tmp)
                
                word_prob = torch.stack(word_prob, dim=0).mean(0)
                return word_prob, enc_hidden

            def collect_active_hidden(inst_beams, inst_idx_to_position_map, enc_hidden, n_bm):
                assert isinstance(enc_hidden, list)
                n_curr_active_inst = len(inst_idx_to_position_map)

                for i in range(len(enc_hidden)):
                    tmp1, tmp2 = enc_hidden[i]
                    _, *d_hs = tmp1.size()
                    
                    new_shape = (n_curr_active_inst * n_bm, *d_hs)
                    tmp1 = tmp1.view(n_curr_active_inst, n_bm, -1)
                    tmp2 = tmp2.view(n_curr_active_inst, n_bm, -1)
                    #print('hidden:', tmp1)

                    for inst_idx, inst_position in inst_idx_to_position_map.items():
                        _prev_ks = inst_beams[inst_idx].get_current_origin()
                        tmp1[inst_position] = tmp1[inst_position].index_select(0, _prev_ks)
                        tmp2[inst_position] = tmp2[inst_position].index_select(0, _prev_ks)
                        #print("PREV_KS:", _prev_ks)
                    #print('after h:', tmp1)
                    tmp1 = tmp1.view(*new_shape)
                    tmp2 = tmp2.view(*new_shape)
                    enc_hidden[i] = (tmp1, tmp2)
                return enc_hidden 

            n_active_inst = len(inst_idx_to_position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams)
            #print(dec_seq)
            #print('before:', enc_hidden[0])
            word_prob, enc_hidden = predict_word(dec_seq, enc_output, enc_hidden, n_active_inst, n_bm, category)
            #print('after:', enc_hidden[0])
            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = self.collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            enc_hidden = collect_active_hidden(inst_dec_beams, inst_idx_to_position_map, enc_hidden, n_bm)
            return active_inst_idx_list, enc_hidden

        with torch.no_grad():
            assert isinstance(enc_output, list)
            assert isinstance(enc_hidden, list)
            assert len(enc_output) == len(self.model)
            assert len(enc_output) == len(enc_hidden)

            for i in range(len(enc_output)):
                if not isinstance(enc_output[i], list):
                    enc_output[i] = [enc_output[i]]

            n_bm = self.opt["beam_size"]
            n_inst, len_s, d_h = enc_output[0][0].size()

            
            #-- Repeat data for beam search
            category = category.unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, self.opt['num_category'])
            enc_output = [[tmp.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h) for tmp in item] for item in enc_output]
            for i in range(len(enc_hidden)):
                if isinstance(enc_hidden[i], tuple):
                    enc_hidden[i] = (enc_hidden[i][0].unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, d_h), 
                        enc_hidden[i][1].unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, d_h))
                else:
                    enc_hidden[i] = enc_hidden[i].unsqueeze(1).repeat(1, n_bm, 1).view(n_inst * n_bm, d_h)

            #-- initialize hidden state
            for i in range(len(enc_output)):
                enc_hidden[i] = self.model[i].decoder.init_hidden(enc_hidden[i])
            

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, self.opt["max_len"], device=self.device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for t in range(1, self.opt["max_len"]):
                active_inst_idx_list, enc_hidden = beam_decode_step(
                    inst_dec_beams, enc_output, enc_hidden, inst_idx_to_position_map, n_bm, category)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                enc_output, enc_hidden, category, inst_idx_to_position_map = self.collate_active_info(
                    enc_output, inst_idx_to_position_map, active_inst_idx_list, category, n_bm, enc_hidden=enc_hidden)

        batch_hyp, batch_scores = self.collect_hypothesis_and_scores(inst_dec_beams, self.opt.get("topk", 1))

        return batch_hyp, batch_scores



    def translate_batch(self, enc_output, enc_hidden, category):
        return self.translate_batch_ENSEMBLE(enc_output, enc_hidden, category)
