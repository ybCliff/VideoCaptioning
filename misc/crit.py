from .logger import AverageMeter
import torch
import torch.nn as nn
import models.Constants as Constants
from torch.autograd import Variable

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def euclidean_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.pow(2).sum(2).t()
    return score  

class BagOfWordsLoss(nn.Module):
    def __init__(self, ignore_index=Constants.PAD):
        super(BagOfWordsLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduce=False)
        self.ignore_index = ignore_index

    def forward(self, scores, target):
        """
        scores: shape of [batch_size, max_len - 1, vocab_size]
        target: shape of [batch_size, max_len - 1]
        """
        assert target.size(1) == scores.size(1)
        shape = scores.shape
        device = scores.device

        mask = target.ne(self.ignore_index).float()
        mask_scores = mask.unsqueeze(-1) * scores
        sum_scores = mask_scores.sum(1)

        #print(sum_scores.shape)
        #print(target.shape)
        labels = Variable(torch.zeros(shape)).to(device)
        labels = labels.scatter_(2, target.unsqueeze(-1), 1).sum(1) # [batch_size, vocab_size]
        labels[:, self.ignore_index] = 0 # pad
        labels[:, Constants.MASK] = 0
        labels = labels.gt(0).float()

        loss = self.loss_fn(sum_scores, labels)

        return torch.sum(loss) / shape[0]

class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.contiguous().view(-1)
        reward = reward.contiguous().view(-1)
        mask = (seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1).cuda(),
                         mask[:, :-1]], 1).contiguous().view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

'''
class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.NLLLoss(ignore_index=ignore_index)

    def forward(self, logits, target):
        """
        logits: shape of [batch_size, max_len - 1, vocab_size], logsoftmax
        target: shape of [batch_size, max_len - 1]
        """
        assert target.size(1) == logits.size(1)
        batch_size = logits.shape[0]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)


        loss = self.loss_fn(logits, target)
        return loss
'''



class TripletLoss(nn.Module):
    """
    triplet ranking loss
    """
    def __init__(self, margin=0.2, measure='cosine', max_violation=True, cost_style='sum'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style

        assert measure in ['order', 'euclidean', 'cosine']
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'euclidean':
            self.sim = euclidean_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward_(self, s, im):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).to(s.device)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_s = cost_s.masked_fill_(I, 0)

        cost_im = (self.margin + scores - d2).clamp(min=0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum()
        else:
            return cost_s.mean() + cost_im.mean()

    def forward(self, embs):
        assert isinstance(embs, list), 'there should be a list of embeddings from the encoder'
        assert len(embs) >= 2, 'there should be at least two kinds of embeddings'
        length = len(embs)
        #embs = [item.mean(1) for item in embs] if len(embs[0].shape) == 3 else embs
        loss = None
        '''
        for i in range(length-1):
            for j in range(i+1, length):
                if loss is None:
                    loss = self.forward_(embs[i], embs[j])
                else:
                    loss += self.forward_(embs[i], embs[j])
        return loss / ((length * (length - 1)) / 2), embs[0].size(0)
        '''
        for emb in embs[0]:
            #print(emb.shape, embs[1].shape)
            for another_emb in embs[1]:
                if loss is None:
                    loss = self.forward_(emb, another_emb)
                else:
                    loss += self.forward_(emb, another_emb)
        return loss / (len(embs[0]) * len(embs[1])), embs[0][0].size(0)

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        logits: shape of [batch_size, max_len - 1, vocab_size]
        target: shape of [batch_size, max_len - 1]
        """
        assert target.size(1) == logits.size(1)
        batch_size = logits.shape[0]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)

        if self.ignore_index is not None:
            mask = target.ne(self.ignore_index).float()

        loss = self.loss_fn(logits, target)

        if self.ignore_index is not None:
            return torch.sum(loss * mask) / batch_size
        return torch.sum(loss) / batch_size

class BDLoss(nn.Module):
    def __init__(self):
        super(BDLoss, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target):
        """
        logits: shape of [batch_size, vocab_size]
        target: shape of [batch_size, 1]
        """
        batch_size = logits.shape[0]
        logits = logits.contiguous().view(-1, logits.shape[-1])
        target = target.contiguous().view(-1)

        weight = target.clone().abs()

        target = target.gt(0).long()


        loss = self.loss_fn(logits, target)

        return torch.sum(loss * weight.float()) / batch_size

class DistillationLoss(nn.Module):
    def __init__(self, ignore_index):
        super(DistillationLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduce=False)
        self.ignore_index = ignore_index

    def forward(self, pred_embs, bert_embs, target):
        """
        logits: shape of [batch_size, max_len - 1, vocab_size]
        target: shape of [batch_size, max_len - 1]
        """
        bsz, seq_len, s = pred_embs.shape
        bert_embs = bert_embs[:, :seq_len, :]
        #print(pred_embs.shape, bert_embs.shape)
        bert_embs = bert_embs.contiguous().view(-1, s)
        pred_embs = pred_embs.contiguous().view(-1, s)
        target = target.contiguous().view(-1)
        #print(bert_embs.shape, pred_embs.shape, target.shape)
        loss = self.loss_fn(pred_embs, bert_embs).sum(-1) 
        #print(loss.shape)
        mask = target.ne(self.ignore_index).float()

        return torch.sum(loss * mask) / (mask.sum() * s)

class AttributeLoss(nn.Module):
    def __init__(self):
        super(AttributeLoss, self).__init__()
        self.loss_fn = nn.BCELoss(reduce=False)

    def forward(self, pred, target):
        """
        logits: shape of [batch_size, max_len - 1, vocab_size]
        target: shape of [batch_size, max_len - 1]
        """

        loss = self.loss_fn(pred, target).sum(-1) / (target.gt(0).sum(-1).float() + 1e-6)
        #print(loss.shape)
        return loss.sum() / target.size(0)

class SelfCritCriterion(nn.Module):
    def __init__(self, keys):
        super(SelfCritCriterion, self).__init__()
        self.keys = keys

    def forward(self, kwargs):
        seq, sample_logprobs, reward, probs = [kwargs[key] for key in self.keys]
        sample_logprobs = sample_logprobs.view(-1)   # (batch_size * max_len)
        reward = reward.view(-1)
        # set mask elements for all <end> tokens to 0 
        mask = (seq>0).float()                        # (batch_size, max_len)
        
        # account for the <end> token in the mask. We do this by shifting the mask one timestep ahead
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).float()
        
        if not mask.is_contiguous():
            mask = mask.contiguous()
        
        mask = mask.view(-1)
        if probs is not None:
            probs = probs.view(-1)
            output = - sample_logprobs * reward * mask * probs
        else:    
            output = - sample_logprobs * reward * mask
        output = torch.sum(output) / seq.size(0) #torch.sum(mask)
        return output

class Criterion(object):
    def __init__(self, 
        crit=[CrossEntropyLoss(ignore_index=Constants.PAD)], 
        keys=[('seq_probs', 'gold')], 
        names=['CAP_LOSS'], 
        scales=[1.0],
        calculate_word_acc=0,
        opt={}
        ):
        assert len(crit) == len(keys)
        assert len(keys) == len(names)
        assert len(names) == len(scales)
        self.num_loss = len(crit)
        self.crit = crit
        self.keys = keys
        self.names = names
        self.scales = scales
        self.calculate_word_acc = calculate_word_acc
        self.calculate_classify_acc = 1 if opt.get('use_beam_decoder', False) else 0
        self.opt = opt

        self.bow_index = -1
        for i, item in enumerate(self.crit):
            if isinstance(item, BagOfWordsLoss):
                self.bow_index = i
                print('BOW index %d' % i)
                break

        self.weights = opt.get('nv_weights', [0.8, 1.0])
        
    def reset_loss_recorder(self):
        # before training a epoch
        self.loss_recorder = [AverageMeter() for _ in range(self.num_loss)]
        self.word_acc_recorder = [AverageMeter() for _ in range(self.calculate_word_acc)]
        self.classify_acc_recorder = [AverageMeter() for _ in range(self.calculate_classify_acc)]

    def check_and_cal(self, pred, gt, crit):
        if isinstance(pred, list):
            i_loss = []
            num_sample = 0
            if isinstance(gt, list):
                assert len(pred) == len(gt)
                if self.opt['method'] in ['nv', 'ag']:
                    assert len(self.weights) == len(gt)
                index = 0
                for p, g in zip(pred, gt):
                    item = crit(p, g)
                    if self.opt['method'] in ['nv', 'ag']:
                        i_loss.append(item * self.weights[index])
                        index += 1
                    else:
                        i_loss.append(item)
                    num_sample += g.size(0)
            else:
                for i in range(len(pred)):
                    i_loss.append(crit(pred[i], gt))
                    num_sample += gt.size(0)
            i_loss = torch.stack(i_loss, dim=0).sum(0)
        else:
            i_loss = crit(pred, gt)
            num_sample = gt.size(0)

        return num_sample, i_loss

    def check_and_cal_word_acc(self, pred, gt):
        assert isinstance(pred, list)
        
        if isinstance(gt, list):
            assert len(pred) == len(gt)
            for i in range(len(pred)):
                ind = gt[i].ne(Constants.PAD)
                if i == 0:
                    if self.opt['method'] == 'signal3':
                        ind = ind & gt[i].ne(self.opt['visual_tag'])
                        #ind = ind & gt[i].ne(Constants.MASK)
                    elif self.opt['method'] == 'ag':
                        ind = ind & gt[i].ne(Constants.BOS)
                    else:
                        ind = ind & gt[i].ne(Constants.MASK)
                elif i == 1 and self.opt['method'] == 'signal3':
                    ind = ind & gt[i].ne(self.opt['nonvisual_tag'])
                    #ind = ind & gt[i].ne(Constants.MASK)
                
                predict_res = pred[i].max(-1)[1][ind]
                target_res = gt[i][ind]

                self.word_acc_recorder[i].update(
                            (predict_res==target_res).sum().item(), 
                            predict_res.size(0), 
                            multiply=False
                    )
        else:
            for i in range(len(pred)):
                ind = gt.ne(Constants.PAD)
                predict_res = pred[i].max(-1)[1][ind]
                target_res = gt[ind]
                self.word_acc_recorder[i].update(
                            (predict_res==target_res).sum().item(), 
                            predict_res.size(0), 
                            multiply=False
                    )

    def cal_classify_acc(self, pred, gt):
        for i, (p, g) in enumerate(zip(pred, gt)):
            self.classify_acc_recorder[i].update(
                            (p.max(dim=-1)[1] == g.long()).sum().item(), 
                            p.size(0), 
                            multiply=False
                    )

    def get_loss(self, results, epoch=-1):
        # input:
        #   - results:  contains the forward results of the model and some ground-truths
        # output:
        #   - loss:     to tune model's parameters
        if epoch != -1:
            rate = min((epoch + 1)/10, 1)
            if self.bow_index != -1:
                self.scales[self.bow_index] = rate

        loss = []
        for i in range(self.num_loss):
            if isinstance(self.crit[i], TripletLoss):
                i_loss, num_sample = self.crit[i](results[self.keys[i]])
            else:
                # prepare the predictions and its corresponding ground-truths
                pred = results[self.keys[i][0]]
                gt = results[self.keys[i][1]]
                #print(gt)
                #print(gt.max())

                # calculate i-th loss
                if isinstance(self.crit[i], DistillationLoss):
                    num_sample = gt.size(0)
                    if self.opt['na']:
                        i_loss = self.crit[i](pred, gt, results['pure_target'])
                    else:
                        i_loss = self.crit[i](pred, gt, results[Constants.mapping['lang'][1]])
                elif isinstance(self.crit[i], SelfCritCriterion):
                    num_sample = pred.size(0)
                    i_loss = self.crit[i](results)
                else:
                    num_sample, i_loss = self.check_and_cal(pred, gt, self.crit[i])

            # weighting i-th loss
            loss.append(i_loss * self.scales[i])

            # update the statistics of i-th loss
            self.loss_recorder[i].update(i_loss.item(), num_sample)

            # For non-autoregressive, calculate accuracy of the generated words
            if self.calculate_word_acc:
                self.check_and_cal_word_acc(results[Constants.mapping['lang'][0]], results[Constants.mapping['lang'][1]])

            if self.calculate_classify_acc:
                self.cal_classify_acc([results[Constants.mapping['beam'][0]]], [results[Constants.mapping['beam'][1]]])

        # loss = loss1 * scale1 + loss2 * scale2 + ...    
        loss = torch.stack(loss, dim=0).sum(0)
        return loss

    def get_loss_info(self):
        # standard operations:
        #   1. before a epoch, Criterion.reset_loss_recorder()
        #   2. during a epoch, Criterion.get_loss(...)
        #   3. after  a epoch, Criterion.get_loss_info()

        loss_info = [meter.avg for meter in self.loss_recorder]
        if self.calculate_word_acc:
            names = self.names + ['Word Acc%d' % i for i in range(self.calculate_word_acc)]
            loss_info = loss_info + [meter.avg for meter in self.word_acc_recorder]
        elif self.calculate_classify_acc:
            names = self.names + ['Classify Acc%d' % i for i in range(self.calculate_classify_acc)]
            loss_info = loss_info + [meter.avg for meter in self.classify_acc_recorder]
        else:
            names = self.names
        return names, loss_info # e.g., CAP_LOSS: 0.1


def get_criterion(opt):
    crit_mapping = {
        'lang': CrossEntropyLoss(ignore_index=Constants.PAD),
        'obj': nn.BCELoss(),
        'ce': CrossEntropyLoss(),
        'tag': CrossEntropyLoss(ignore_index=Constants.PAD),
        'length': nn.SmoothL1Loss() if not opt.get('use_kl', False) else nn.KLDivLoss(),
        'bow': BagOfWordsLoss(ignore_index=Constants.PAD),
        'attr': nn.BCELoss(),
        'attr2': AttributeLoss(),
        'dist': DistillationLoss(ignore_index=Constants.PAD), # bert distillation
        'beam': BDLoss(), #nn.NLLLoss(),  # beam candidate classification
        'self_crit': SelfCritCriterion(Constants.mapping['self_crit']),   # self-crit reinforcement learning
        'triplet': TripletLoss(),
    }
    if opt.get('bow_loss', False):
        opt['crit'].append('bow')
        opt['crit_name'].append('BOW Loss')
        opt['crit_key'].append(Constants.mapping['bow'])
        opt['crit_scale'].append(0.1)

    crit_type = opt['crit']
    assert isinstance(crit_type, list)
    
    crit = []
    for item in crit_type:
        assert item.lower() in crit_mapping
        crit.append(crit_mapping[item.lower()])

    if opt['na']:
        if opt['method'] == 'mp':
            calculate_word_acc = 1
        elif opt['method'] == 'direct':
            calculate_word_acc = 3
        elif opt['method'] == 'signal':
            calculate_word_acc = 3
        elif opt['method'] == 'signal3':
            calculate_word_acc = 3
        elif opt['method'] == 'signal2':
            calculate_word_acc = 2
        elif opt['method'] == 'nv':
            calculate_word_acc = 2
        elif opt['method'] == 'ms':
            calculate_word_acc = 2
    else:
        if opt['method'] == 'ag':
            calculate_word_acc = 2 if (not opt.get('use_beam_decoder', False) and not opt.get('use_rl', False)) else 0
        else:
            calculate_word_acc = 0

    return Criterion(
            crit=crit,
            keys=opt['crit_key'],
            names=opt['crit_name'],
            scales=opt['crit_scale'],
            calculate_word_acc=calculate_word_acc,
            opt=opt
        )
