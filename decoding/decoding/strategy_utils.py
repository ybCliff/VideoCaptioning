import torch
import torch.nn.functional as F


def duplicate_encoder_out(encoder_out, bsz, beam_size):
    encoder_out['encoder_out'] = encoder_out['encoder_out'].unsqueeze(2).repeat(1, 1, beam_size, 1).view(-1, bsz * beam_size, encoder_out['encoder_out'].size(-1))
    if encoder_out['encoder_padding_mask'] is not None:
        encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].unsqueeze(1).repeat(1, beam_size, 1).view(bsz * beam_size, -1)


def generate_step_with_prob(out):
    probs = F.softmax(out, dim=-1)
    max_probs, idx = probs.max(dim=-1)
    return idx, max_probs, probs


def assign_single_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y


def assign_multi_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y.view(-1)[i.view(-1).nonzero()]


def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y


def assign_multi_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y.view(-1)[i.view(-1)]


def convert_tokens(dictionary, tokens):
    return ' '.join([dictionary[str(token)] for token in tokens])