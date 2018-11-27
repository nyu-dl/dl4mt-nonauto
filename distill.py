import copy
import math
import os
import torch
import numpy as np
import time

from torch.nn import functional as F
from torch.autograd import Variable

from tqdm import tqdm, trange
from model import Transformer, FastTransformer, INF, TINY, softmax
from data import NormalField, NormalTranslationDataset, TripleTranslationDataset, ParallelDataset, data_path
from utils import Metrics, Best, computeBLEU, Batch, masked_sort, computeGroupBLEU, organise_trg_len_dic, make_decoder_masks, double_source_masks, remove_repeats, remove_repeats_tensor, print_bleu
from time import gmtime, strftime
import copy
import json

tokenizer = lambda x: x.replace('@@ ', '').split()

def distill_model(args, model, dev, evaluate=True,
                distill_path=None, names=None, maxsteps=None):

    if not args.no_tqdm:
        progressbar = tqdm(total=200, desc='start decoding')

    trg_len_dic = None

    args.logger.info("decoding, f_size={}, beam_size={}, alpha={}".format(args.f_size, args.beam_size, args.alpha))
    dev.train = False  # make iterator volatile=True

    model.eval()
    if distill_path is not None:
        if args.dataset != "mscoco":
            handles = [open(os.path.join(distill_path, name), 'w') for name in names]
        else:
            distill_annots = []
            distill_filepath = os.path.join(str(distill_path), "train.bpe.fixed.distill")


    corpus_size = 0
    src_outputs, trg_outputs, dec_outputs, timings = [], [], [], []
    all_decs = [ [] for idx in range(args.valid_repeat_dec)]
    decoded_words, target_words, decoded_info = 0, 0, 0

    attentions = None
    decoder = model.decoder[0] if args.model is FastTransformer else model.decoder
    pad_id = decoder.field.vocab.stoi['<pad>']
    eos_id = decoder.field.vocab.stoi['<eos>']

    curr_time = 0
    cum_bs = 0

    for iters, dev_batch in enumerate(dev):

        start_t = time.time()

        if args.dataset != "mscoco":
            decoder_inputs, decoder_masks,\
            targets, target_masks,\
            sources, source_masks,\
            encoding, batch_size, rest = model.quick_prepare(dev_batch, fast=(type(model) is FastTransformer), trg_len_option=args.trg_len_option, trg_len_ratio=args.trg_len_ratio, trg_len_dic=trg_len_dic, bp=args.bp)
        else:
            all_captions = dev_batch[1]
            all_img_names = dev_batch[2]
            dev_batch[1] = dev_batch[1][0]
            decoder_inputs, decoder_masks,\
            targets, target_masks,\
            _, source_masks,\
            encoding, batch_size, rest = model.quick_prepare_mscoco(dev_batch, all_captions=all_captions, fast=(type(model) is FastTransformer), inputs_dec=args.inputs_dec, trg_len_option=args.trg_len_option, max_len=args.max_offset, trg_len_dic=trg_len_dic, bp=args.bp)


        corpus_size += batch_size

        batch_size, src_len, hsize = encoding[0].size()

        with torch.no_grad():
            # for now
            if type(model) is Transformer:
                all_decodings = []
                decoding = model(encoding, source_masks, decoder_inputs, decoder_masks,
                                beam=args.beam_size, alpha=args.alpha, \
                                 decoding=True, feedback=attentions)
                all_decodings.append( decoding )
                curr_iter = [0]

        used_t = time.time() - start_t
        curr_time += used_t

        real_mask = 1 - ((decoding.data == eos_id) + (decoding.data == pad_id)).float()
        if args.dataset != "mscoco":
            outputs = [model.output_decoding(d, False) for d in [('src', sources), ('trg', targets), ('trg', decoding)]]

            for s, t, d in zip(outputs[0], outputs[1], outputs[-1]):
                #s, t, d = s.replace('@@ ', ''), t.replace('@@ ', ''), d.replace('@@ ', '')
                print(s, file=handles[0], flush=True)
                print(t, file=handles[1], flush=True)
                print(d, file=handles[2], flush=True)
        else:
            outputs = [model.output_decoding(d, unbpe=False) for d in [('trg', targets), ('trg', decoding)]]

            for c, (t, d) in enumerate(zip(outputs[0], outputs[1])):
                annot = {}
                annot['bpes'] = [d]
                annot['img_name'] = all_img_names[c]
                distill_annots.append(annot)

            json.dump(distill_annots, open(distill_filepath, 'w'))

        if not args.no_tqdm:
            progressbar.update(iters)
            progressbar.set_description('finishing sentences={}/batches={}, \
                length={}/average iter={}, speed={} sec/batch'.format(\
                corpus_size, iters, src_len, np.mean(np.array(curr_iter)), curr_time / (1 + iters)))

    if args.dataset == "mscoco":
        json.dump(distill_annots, open(distill_filepath, 'w'))

    args.logger.info("Total time {}".format((curr_time / float(cum_bs) * 1000)))
