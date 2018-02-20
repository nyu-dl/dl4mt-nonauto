import math
import ipdb
import torch
import random
import numpy as np
import _pickle as pickle
import revtok
import os
from itertools import groupby
import getpass
from collections import Counter

from torch.autograd import Variable
from torchtext import data, datasets
from nltk.translate.gleu_score import sentence_gleu, corpus_gleu
from nltk.translate.bleu_score import closest_ref_length, brevity_penalty, modified_precision, SmoothingFunction
from contextlib import ExitStack
from collections import OrderedDict
import fractions

import torchvision

try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction

def sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25),
                  smoothing_function=None, auto_reweigh=False,
                  emulate_multibleu=False):

    return corpus_bleu([references], [hypothesis],
                        weights, smoothing_function, auto_reweigh,
                        emulate_multibleu)


def corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=None, auto_reweigh=False,
                emulate_multibleu=False):
    p_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    if len(list_of_references) != len(hypotheses):
        print ("The number of hypotheses and their reference(s) should be the same")
        return (0, (0, 0, 0, 0), 0, 0, 0)

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len =  len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Uniformly re-weighting based on maximum hypothesis lengths if largest
    # order of n-grams < 4 and weights is set at default.
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = ( 1 / hyp_lengths ,) * hyp_lengths

    # Collects the various precision values for the different ngram orders.
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(weights, start=1)]

    p_n_ = [xx.numerator / xx.denominator * 100 for xx in p_n]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return (0, (0, 0, 0, 0), 0, 0, 0)

    # If there's no smoothing, set use method0 from SmoothinFunction class.
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    p_n = smoothing_function(p_n, references=references, hypothesis=hypothesis,
                             hyp_len=hyp_len, emulate_multibleu=emulate_multibleu)
    s = (w * math.log(p_i) for i, (w, p_i) in enumerate(zip(weights, p_n)))
    s =  bp * math.exp(math.fsum(s)) * 100
    final_bleu = round(s, 4) if emulate_multibleu else s
    return (final_bleu, p_n_, bp, ref_lengths, hyp_lengths)

INF = 1e10
TINY = 1e-9
def computeGLEU(outputs, targets, corpus=False, tokenizer=None):
    if tokenizer is None:
        tokenizer = revtok.tokenize

    outputs = [tokenizer(o) for o in outputs]
    targets = [tokenizer(t) for t in targets]

    if not corpus:
        return torch.Tensor([sentence_gleu(
            [t],  o) for o, t in zip(outputs, targets)])
    return corpus_gleu([[t] for t in targets], [o for o in outputs])

def computeBLEU(outputs, targets, corpus=False, tokenizer=None):
    if tokenizer is None:
        tokenizer = revtok.tokenize

    outputs = [tokenizer(o) for o in outputs]
    targets = [tokenizer(t) for t in targets]

    if corpus:
        return corpus_bleu([[t] for t in targets], [o for o in outputs], emulate_multibleu=True)
    else:
        return [sentence_bleu([t],  o)[0] for o, t in zip(outputs, targets)]
        #return torch.Tensor([sentence_bleu([t],  o)[0] for o, t in zip(outputs, targets)])

def computeBLEUMSCOCO(outputs, targets, corpus=True, tokenizer=None):
    # outputs is list of 5000 captions
    # targets is list of 5000 lists each length of 5
    if tokenizer is None:
        tokenizer = revtok.tokenize

    outputs = [tokenizer(o) for o in outputs]
    new_targets = []
    for i, t in enumerate(targets):
        new_targets.append([tokenizer(tt) for tt in t])
        #targets[i] = [tokenizer(tt) for tt in t]

    if corpus:
        return corpus_bleu(new_targets, outputs, emulate_multibleu=True)
    else:
        return [sentence_bleu(new_t, o)[0] for o, new_t in zip(outputs, new_targets)]

def compute_bp(hypotheses, list_of_references):
    hyp_lengths, ref_lengths = 0, 0
    for references, hypothesis in zip(list_of_references, hypotheses):
        hyp_len =  len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)
    return bp

def computeGroupBLEU(outputs, targets, tokenizer=None, bra=10, maxmaxlen=80):
    if tokenizer is None:
        tokenizer = revtok.tokenize

    outputs = [tokenizer(o) for o in outputs]
    targets = [tokenizer(t) for t in targets]
    maxlens = max([len(t) for t in targets])
    print(maxlens)
    maxlens = min([maxlens, maxmaxlen])
    nums = int(np.ceil(maxlens / bra))
    outputs_buckets = [[] for _ in range(nums)]
    targets_buckets = [[] for _ in range(nums)]
    for o, t in zip(outputs, targets):
        idx = len(o) // bra
        if idx >= len(outputs_buckets):
            idx = -1
        outputs_buckets[idx] += [o]
        targets_buckets[idx] += [t]

    for k in range(nums):
        print(corpus_bleu([[t] for t in targets_buckets[k]], [o for o in outputs_buckets[k]], emulate_multibleu=True))

class TargetLength:
    def __init__(self, lengths=None): # data_type : sum, avg
        self.lengths = lengths if lengths != None else dict()

    def accumulate(self, batch):
        src_len = (batch.src != 1).sum(-1).cpu().data.numpy()
        trg_len = (batch.trg != 1).sum(-1).cpu().data.numpy()
        for (slen, tlen) in zip(src_len, trg_len):
            if not slen in self.lengths:
                self.lengths[slen] = (1, int(tlen))
            else:
                (count, acc) = self.lengths[slen]
                self.lengths[slen] = (count + 1, acc + int(tlen))

    def get_trg_len(self, src_len):
        if not src_len in self.lengths:
            return self.get_trg_len(src_len + 1) - 1
        else:
            (count, acc) = self.lengths[src_len]
            return acc / float(count)

def organise_trg_len_dic(trg_len_dic):
    trg_len_dic = {k:int(v[1]/float(v[0])) for (k, v) in trg_len_dic.items()}
    return trg_len_dic

def query_trg_len_dic(trg_len_dic, q):
    max_src_len = max(trg_len_dic.keys())
    if q <= max_src_len:
        if q in trg_len_dic:
            return trg_len_dic[q]
        else:
            return query_trg_len_dic(trg_len_dic, q+1) - 1
    else:
        return int(math.floor( trg_len_dic[max_src_len] / max_src_len * q ))

def make_decoder_masks(source_masks, trg_len_dic):
    batch_size, src_max_len = source_masks.size()
    src_len = (source_masks == 1).sum(-1).cpu().numpy()
    trg_len = [int(math.floor(query_trg_len_dic(trg_len_dic, src) * 1.1)) for src in src_len]
    trg_max_len = max(trg_len)
    decoder_masks = np.zeros((batch_size, trg_max_len))
    #decoder_masks = Variable(torch.zeros(batch_size, trg_max_len), requires_grad=False)
    for idx, tt in enumerate(trg_len):
        decoder_masks[idx][:tt] = 1
    result = torch.from_numpy(decoder_masks).float()
    if source_masks.is_cuda:
        result = result.cuda()
    return result

def double_source_masks(source_masks):
    batch_size, src_max_len = source_masks.size()
    src_len = (source_masks == 1).sum(-1).cpu().numpy()
    decoder_masks = np.zeros((batch_size, src_max_len * 2))
    for idx, tt in enumerate(src_len):
        decoder_masks[idx][:2*tt] = 1
    result = torch.from_numpy(decoder_masks).float()
    if source_masks.is_cuda:
        result = result.cuda()
    return result

class Metrics:

    def __init__(self, name, *metrics, data_type="sum"): # data_type : sum, avg
        self.count = 0
        self.metrics = OrderedDict((metric, 0) for metric in metrics)
        self.name = name
        self.data_type = data_type

    def accumulate(self, count, *values, print_iter=None):
        self.count += count
        if print_iter is not None:
            print(print_iter, end=' ')
        for value, metric in zip(values, self.metrics):
            if isinstance(value, torch.autograd.Variable):
                value = value.data
            if torch.is_tensor(value):
                with torch.cuda.device_of(value):
                    value = value.cpu()
                value = value.float().sum()

            if print_iter is not None:
                print('%.3f' % value, end=' ')
            if self.data_type == "sum":
                self.metrics[metric] += value
            elif self.data_type == "avg":
                self.metrics[metric] += value * count

        if print_iter is not None:
            print()
        return values[0] # loss

    def __getattr__(self, key):
        if key in self.metrics:
            return self.metrics[key] / (self.count + 1e-9)
        raise AttributeError

    def __repr__(self):
        return ("{}: ".format(self.name) +
               "[{}]".format( ', '.join(["{:.4f}".format(getattr(self, metric)) for metric, value in self.metrics.items() if value is not 0 ] ) ) )

    def tensorboard(self, expt, i):
        for metric in self.metrics:
            value = getattr(self, metric)
            if value != 0:
                #expt.add_scalar_value(f'{self.name}_{metric}', value, step=i)
                expt.add_scalar_value("{}_{}".format(self.name, metric), value, step=i)

    def reset(self):
        self.count = 0
        self.metrics.update({metric: 0 for metric in self.metrics})

class Best:
    def __init__(self, cmp_fn, *metrics, model=None, opt=None, path='', gpu=0, which=[0]):
        self.cmp_fn = cmp_fn
        self.model = model
        self.opt = opt
        self.path = path + '.pt'
        self.metrics = OrderedDict((metric, None) for metric in metrics)
        self.gpu = gpu
        self.which = which
        self.best_cmp_value = None

    def accumulate(self, *other_values):

        with torch.cuda.device(self.gpu):
            cmp_values = [other_values[which] for which in self.which]
            if self.best_cmp_value is None or \
               self.cmp_fn(self.best_cmp_value, *cmp_values) != self.best_cmp_value:
                self.metrics.update( { metric: value for metric, value in zip(
                    list(self.metrics.keys()), other_values) } )
                self.best_cmp_value = self.cmp_fn( [ list(self.metrics.items())[which][1] for which in self.which ] )

                #open(self.path + '.temp', 'w')
                if self.model is not None:
                    torch.save(self.model.state_dict(), self.path)

                if self.opt is not None:
                    torch.save([self.i, self.opt.state_dict()], self.path + '.states')
                #os.remove(self.path + '.temp')

    def __getattr__(self, key):
        if key in self.metrics:
            return self.metrics[key]
        raise AttributeError

    def __repr__(self):
        return ("BEST: " +
                ', '.join(["{}: {:.4f}".format(metric, getattr(self, metric)) for metric, value in self.metrics.items() if value is not 0]))

class CacheExample(data.Example):

    @classmethod
    def fromsample(cls, data_lists, names):
        ex = cls()
        for data, name in zip(data_lists, names):
            setattr(ex, name, data)
        return ex


class Cache:

    def __init__(self, size=10000, fileds=["src", "trg"]):
        self.cache = []
        self.maxsize = size

    def demask(self, data, mask):
        with torch.cuda.device_of(data):
            data = [d[:l] for d, l in zip(data.data.tolist(), mask.sum(1).long().tolist())]
        return data

    def add(self, data_lists, masks, names):
        data_lists = [self.demask(d, m) for d, m in zip(data_lists, masks)]
        for data in zip(*data_lists):
            self.cache.append(CacheExample.fromsample(data, names))

        if len(self.cache) >= self.maxsize:
            self.cache = self.cache[-self.maxsize:]


class Batch:
    def __init__(self, src=None, trg=None, dec=None):
        self.src, self.trg, self.dec = src, trg, dec

def masked_sort(x, mask, dim=-1):
    x.data += ((1 - mask) * INF).long()
    y, i = torch.sort(x, dim)
    y.data *= mask.long()
    return y, i

def unsorted(y, i, dim=-1):
    z = Variable(y.data.new(*y.size()))
    z.scatter_(dim, i, y)
    return z


def merge_cache(decoding_path, names0, last_epoch=0, max_cache=20):
    file_lock = open(decoding_path + '/_temp_decode', 'w')

    for name in names0:
        filenames = []
        for i in range(max_cache):
            filenames.append('{}/{}.ep{}'.format(decoding_path, name, last_epoch - i))
            if (last_epoch - i) <= 0:
                break
        code = 'cat {} > {}.train.{}'.format(" ".join(filenames), '{}/{}'.format(decoding_path, name), last_epoch)
        os.system(code)
    os.remove(decoding_path + '/_temp_decode')

def corrupt_target_fix(trg, decoder_masks, vocab_size, weight=0.1, cor_p=[0.1, 0.1, 0.1, 0.1]):
    batch_size, max_trg_len = trg.size() # actual trg len
    max_dec_len = decoder_masks.size(1) # 2 * actual src len
    dec_lens = (decoder_masks == 1).sum(-1).cpu().numpy()
    trg_lens = (trg != 1).sum(-1).data.cpu().numpy()

    num_corrupts = np.array( [ np.random.choice(dec_lens[bidx]//2,
                                               min( max( math.floor(weight * (dec_lens[bidx]//2)), 1 ), dec_lens[bidx]//2),
                                               replace=False ) \
                             for bidx in range(batch_size) ] )

    #min_len = min(max_trg_len, max_dec_len)
    decoder_input = np.ones((batch_size, max_dec_len))
    decoder_input.fill(3)
    #decoder_input[:, :min_len] = trg[:, :min_len].data.cpu().numpy()

    for bidx in range(batch_size):
        min_len = min(dec_lens[bidx], trg_lens[bidx])
        decoder_input[bidx][:min_len] = trg[bidx, :min_len].data.cpu().numpy()
        nr_list = num_corrupts[bidx]
        for nr in nr_list:

            prob = np.random.rand()

            #### each corruption changes multiple words
            if prob < sum(cor_p[:1]): # repeat
                decoder_input[bidx][nr+1:] = decoder_input[bidx][nr:-1]

            elif prob < sum(cor_p[:2]): # drop
                decoder_input[bidx][nr:-1] = decoder_input[bidx][nr+1:]

            #### each corruption changes one word
            elif prob < sum(cor_p[:3]): # replace word with random word
                decoder_input[bidx][nr] = np.random.randint(vocab_size-4) + 4

            #### each corruption changes two words
            elif prob < sum(cor_p[:4]): # swap
                temp = decoder_input[bidx][nr]
                decoder_input[bidx][nr] = decoder_input[bidx][nr+1]
                decoder_input[bidx][nr+1] = temp

    result = torch.from_numpy(decoder_input).long()
    if decoder_masks.is_cuda:
        result = result.cuda(decoder_masks.get_device())
    return Variable(result, requires_grad=False)

def corrupt_target(trg, decoder_masks, vocab_size, weight=0.1, cor_p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    batch_size, max_trg_len = trg.size()
    max_dec_len = decoder_masks.size(1)
    dec_lens = (decoder_masks == 1).sum(-1).cpu().numpy()

    num_corrupts = np.array( [ np.random.choice(dec_lens[bidx]-1,
                                               min( max( math.floor(weight * dec_lens[bidx]), 1 ), dec_lens[bidx]-1 ),
                                               replace=False ) \
                             for bidx in range(batch_size) ] )

    min_len = min(max_trg_len, max_dec_len)
    decoder_input = np.ones((batch_size, max_dec_len))
    decoder_input.fill(3)
    decoder_input[:, :min_len] = trg[:, :min_len].data.cpu().numpy()

    for bidx in range(batch_size):
        nr_list = num_corrupts[bidx]
        for nr in nr_list:

            prob = np.random.rand()

            #### each corruption changes multiple words
            if prob < sum(cor_p[:1]): # repeat
                decoder_input[bidx][nr+1:] = decoder_input[bidx][nr:-1]

            elif prob < sum(cor_p[:2]): # drop
                decoder_input[bidx][nr:-1] = decoder_input[bidx][nr+1:]

            elif prob < sum(cor_p[:3]): # add random word
                decoder_input[bidx][nr+1:] = decoder_input[bidx][nr:-1]
                decoder_input[bidx][nr] = np.random.randint(vocab_size-4) + 4 # sample except UNK/PAD/INIT/EOS

            #### each corruption changes one word
            elif prob < sum(cor_p[:4]): # repeat and drop next
                decoder_input[bidx][nr+1] = decoder_input[bidx][nr]

            elif prob < sum(cor_p[:5]): # replace word with random word
                decoder_input[bidx][nr] = np.random.randint(vocab_size-4) + 4

            #### each corruption changes two words
            elif prob < sum(cor_p[:6]): # swap
                temp = decoder_input[bidx][nr]
                decoder_input[bidx][nr] = decoder_input[bidx][nr+1]
                decoder_input[bidx][nr+1] = temp

            elif prob < sum(cor_p[:7]): # global swap
                swap_idx = np.random.randint(1, dec_lens[bidx]-nr) + nr
                temp = decoder_input[bidx][nr]
                decoder_input[bidx][nr] = decoder_input[bidx][swap_idx]
                decoder_input[bidx][swap_idx] = temp

    result = torch.from_numpy(decoder_input).long()
    if decoder_masks.is_cuda:
        result = result.cuda(decoder_masks.get_device())
    return Variable(result, requires_grad=False)

def drop(sentence, n_d):
    cur_len = np.sum( sentence != 1 )
    for idx in range(n_d):
        drop_pos = random.randint(0, cur_len - 1) # a <= N <= b
        sentence[drop_pos:-1] = sentence[drop_pos+1:]
        cur_len = cur_len - 1
    sentence[-n_d:] = 1
    return sentence

def repeat(sentence, n_r):
    cur_len = np.sum( sentence != 1 )
    for idx in range(n_r):
        drop_pos = random.randint(0, cur_len) # a <= N <= b
        sentence[drop_pos+1:] = sentence[drop_pos:-1]
    sentence[cur_len:] = 1
    return sentence

def remove_repeats(lst_of_sentences):
    lst = []
    for sentence in lst_of_sentences:
        lst.append( " ".join([x[0] for x in groupby(sentence.split())]) )
    return lst

def remove_repeats_tensor(tensor):
    tensor = tensor.data.cpu()
    newtensor = tensor.clone()
    batch_size, seq_len = tensor.size()
    for bidx in range(batch_size):
        for sidx in range(seq_len-1):
            if newtensor[bidx, sidx] == newtensor[bidx, sidx+1]:
                newtensor[bidx, sidx:-1] = newtensor[bidx, sidx+1:]
    return Variable(newtensor)

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def print_bleu(bleu_output, verbose=True):
    (final_bleu, prec, bp, ref_lengths, hyp_lengths) = bleu_output
    ratio = 0 if ref_lengths == 0 else hyp_lengths/ref_lengths
    if verbose:
        return "BLEU = {:.2f}, {:.1f}/{:.1f}/{:.1f}/{:.1f} (BP={:.3f}, ratio={:.3f}, hyp_len={}, ref_len={})".format(
            final_bleu, prec[0], prec[1], prec[2], prec[3], bp, ratio, hyp_lengths, ref_lengths
        )
    else:
        return "BLEU = {:.2f}, {:.1f}/{:.1f}/{:.1f}/{:.1f} (BP={:.3f}, ratio={:.3f})".format(
            final_bleu, prec[0], prec[1], prec[2], prec[3], bp, ratio
        )

def set_eos(argmax):
    new_argmax = Variable(argmax.data.new(*argmax.size()), requires_grad=False)
    new_argmax.fill_(3)
    batch_size, seq_len = argmax.size()
    argmax_lst = argmax.data.cpu().numpy().tolist()
    for bidx in range(batch_size):
        if 3 in argmax_lst[bidx]:
            idx = argmax_lst[bidx].index(3)
            if idx > 0 :
                new_argmax[bidx,:idx] = argmax[bidx,:idx]
    return new_argmax

def init_encoder(model, saved):
    saved_ = {k.replace("encoder.",""):v for (k,v) in saved.items() if "encoder" in k}
    encoder = model.encoder
    encoder.load_state_dict(saved_)
    return model

def oracle_converged(bleu_hist, num_items=5):
    batch_size = len(bleu_hist)
    converged = [False for bidx in range(batch_size)]
    for bidx in range(batch_size):
        if len(bleu_hist[bidx]) < num_items:
            converged[bidx] = False
        else:
            converged[bidx] = True
            hist = bleu_hist[bidx][-num_items:]
            for item in hist[1:]:
                if item > hist[0]:
                    converged[bidx] = False # if BLEU improves in 4 iters, not converged
    return converged

def equality_converged(output_hist, num_items=5):
    batch_size = len(output_hist)
    converged = [False for bidx in range(batch_size)]
    for bidx in range(batch_size):
        if len(output_hist[bidx]) < num_items:
            converged[bidx] = False
        else:
            converged[bidx] = False
            hist = output_hist[bidx][-num_items:]
            for item in hist[1:]:
                if item == hist[0]:
                    converged[bidx] = True # if out_i == out_j for (j = i+1, i+2, i+3, i+4), converged
    return converged

def jaccard_converged(multiset_hist, num_items=5, jaccard_thresh=1.0):
    batch_size = len(multiset_hist)
    converged = [False for bidx in range(batch_size)]
    for bidx in range(batch_size):
        if len(multiset_hist[bidx]) < num_items:
            converged[bidx] = False
        else:
            converged[bidx] = False
            hist = multiset_hist[bidx][-num_items:]
            for item in hist[1:]:

                inters = len(item.intersection(hist[0]))
                unio = len(item.union(hist[0]))
                jaccard_index = float(inters) / np.maximum(1.,float(unio))

                if jaccard_index >= jaccard_thresh:
                    converged[bidx] = True
    return converged
