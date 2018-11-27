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
from utils import Metrics, Best, computeBLEU, computeBLEUMSCOCO, Batch, masked_sort, computeGroupBLEU, organise_trg_len_dic, make_decoder_masks, \
        double_source_masks, remove_repeats, remove_repeats_tensor, print_bleu, oracle_converged, equality_converged, jaccard_converged
from time import gmtime, strftime
import copy
try:
    from multiset import Multiset
except:
    pass

tokenizer = lambda x: x.replace('@@ ', '').split()

def run_fast_transformer(decoder_inputs, decoder_masks,\
                        sources, source_masks,\
                        targets,\
                        encoding,\
                        model, args, use_argmax=True):

    trg_unidx = model.output_decoding( ('trg', targets) )

    batch_size, src_len, hsize = encoding[0].size()

    all_decodings = []
    all_probs = []
    iter_ = 0
    bleu_hist = [ [] for xx in range(batch_size) ]
    output_hist = [ [] for xx in range(batch_size) ]
    multiset_hist = [ [] for xx in range(batch_size) ]
    num_iters = [ 0 for xx in range(batch_size) ]
    done_ = [False for xx in range(batch_size)]
    final_decoding = [ None for xx in range(batch_size) ]

    while True:
        curr_iter = min(iter_, args.num_decs-1)
        next_iter = min(iter_+1, args.num_decs-1)

        decoding, out, probs = model(encoding, source_masks, decoder_inputs, decoder_masks,
                                     decoding=True, return_probs=True, iter_=curr_iter)

        dec_output = decoding.data.cpu().numpy().tolist()

        """
        if args.trg_len_option != "reference":
            decoder_masks = 0. * decoder_masks
            for bidx in range(batch_size):
                try:
                    decoder_masks[bidx,:(dec_output[bidx].index(3))+1] = 1.
                except:
                    decoder_masks[bidx,:] = 1.
        """

        if args.adaptive_decoding == "oracle":
            out_unidx = model.output_decoding( ('trg', decoding ) )
            sentence_bleus = computeBLEU(out_unidx, trg_unidx, corpus=False, tokenizer=tokenizer)

            for bidx in range(batch_size):
                output_hist[bidx].append( dec_output[bidx] )
                bleu_hist[bidx].append(sentence_bleus[bidx])

            converged = oracle_converged( bleu_hist, num_items=args.adaptive_window )
            for bidx in range(batch_size):
                if not done_[bidx] and converged[bidx] and num_iters[bidx] == 0:
                    num_iters[bidx] = iter_ + 1 - (args.adaptive_window -1)
                    done_[bidx] = True
                    final_decoding[bidx] = output_hist[bidx][-args.adaptive_window]

        elif args.adaptive_decoding == "equality":
            for bidx in range(batch_size):
                #if 3 in dec_output[bidx]:
                #    dec_output[bidx] = dec_output[bidx][:dec_output[bidx].index(3)]
                output_hist[bidx].append( dec_output[bidx] )

            converged = equality_converged( output_hist, num_items=args.adaptive_window )

            for bidx in range(batch_size):
                if not done_[bidx] and converged[bidx] and num_iters[bidx] == 0:
                    num_iters[bidx] = iter_ + 1
                    done_[bidx] = True
                    final_decoding[bidx] = output_hist[bidx][-1]

        elif args.adaptive_decoding == "jaccard":
            for bidx in range(batch_size):
                #if 3 in dec_output[bidx]:
                #    dec_output[bidx] = dec_output[bidx][:dec_output[bidx].index(3)]
                output_hist[bidx].append( dec_output[bidx] )
                multiset_hist[bidx].append( Multiset(dec_output[bidx]) )

            converged = jaccard_converged( multiset_hist, num_items=args.adaptive_window )

            for bidx in range(batch_size):
                if not done_[bidx] and converged[bidx] and num_iters[bidx] == 0:
                    num_iters[bidx] = iter_ + 1
                    done_[bidx] = True
                    final_decoding[bidx] = output_hist[bidx][-1]

        all_decodings.append( decoding )
        all_probs.append(probs)

        decoder_inputs = 0
        if args.next_dec_input in ["both", "emb"]:
            if use_argmax:
                _, argmax = torch.max(probs, dim=-1)
            else:
                probs_sz = probs.size()
                probs_ = Variable(probs.data, requires_grad=False)
                argmax = torch.multinomial(probs_.contiguous().view(-1, probs_sz[-1]), 1).view(*probs_sz[:-1])
            emb = F.embedding(argmax, model.decoder[next_iter].out.weight * math.sqrt(args.d_model))
            decoder_inputs += emb

        if args.next_dec_input in ["both", "out"]:
            decoder_inputs += out

        iter_ += 1
        if iter_ == args.valid_repeat_dec or (False not in done_):
            break

    if args.adaptive_decoding != None:
        for bidx in range(batch_size):
            if num_iters[bidx] == 0:
                num_iters[bidx] = 20
            if final_decoding[bidx] == None:
                if args.adaptive_decoding == "oracle":
                    final_decoding[bidx] = output_hist[bidx][np.argmax(bleu_hist[bidx])]
                else:
                    final_decoding[bidx] = output_hist[bidx][-1]

        decoding = Variable(torch.LongTensor(np.array(final_decoding)))
        if decoder_masks.is_cuda:
            decoding = decoding.cuda()

    return decoding, all_decodings, num_iters, all_probs

def decode_model(args, model, dev, evaluate=True, trg_len_dic=None,
                decoding_path=None, names=None, maxsteps=None):

    args.logger.info("decoding, f_size={}, beam_size={}, alpha={}".format(args.f_size, args.beam_size, args.alpha))
    dev.train = False  # make iterator volatile=True

    if not args.no_tqdm:
        progressbar = tqdm(total=200, desc='start decoding')

    model.eval()
    if args.save_decs:
        decoding_path.mkdir(parents=True, exist_ok=True)
        handles = [(decoding_path / name ).open('w') for name in names]

    corpus_size = 0
    src_outputs, trg_outputs, dec_outputs, timings = [], [], [], []
    all_decs = [ [] for idx in range(args.valid_repeat_dec)]
    decoded_words, target_words, decoded_info = 0, 0, 0

    attentions = None
    decoder = model.decoder[0] if args.model is FastTransformer else model.decoder
    pad_id = decoder.field.vocab.stoi['<pad>']
    eos_id = decoder.field.vocab.stoi['<eos>']

    curr_time = 0
    cum_sentences = 0
    cum_tokens = 0
    cum_images = 0 # used for mscoco
    num_iters_total = []

    for iters, dev_batch in enumerate(dev):
        start_t = time.time()

        if args.dataset != "mscoco":
            if args.trg_len_option == "predict":
                _, _, decoder_inputs, decoder_masks,\
                targets, target_masks,\
                sources, source_masks,\
                encoding, batch_size, rest = model.quick_prepare(dev_batch, fast=(type(model) is FastTransformer), trg_len_option=args.trg_len_option, trg_len_ratio=args.trg_len_ratio, trg_len_dic=trg_len_dic, bp=args.bp)
            else:
                decoder_inputs, decoder_masks,\
                targets, target_masks,\
                sources, source_masks,\
                encoding, batch_size, rest = model.quick_prepare(dev_batch, fast=(type(model) is FastTransformer), trg_len_option=args.trg_len_option, trg_len_ratio=args.trg_len_ratio, trg_len_dic=trg_len_dic, bp=args.bp)

        else:
            # only use first caption for calculating log likelihood
            all_captions = dev_batch[1]
            dev_batch[1] = dev_batch[1][0]
            decoder_inputs, decoder_masks,\
            targets, target_masks,\
            _, source_masks,\
            encoding, batch_size, rest = model.quick_prepare_mscoco(dev_batch, all_captions=all_captions, fast=(type(model) is FastTransformer), inputs_dec=args.inputs_dec, trg_len_option=args.trg_len_option, max_len=args.max_len, trg_len_dic=trg_len_dic, bp=args.bp, gpu=args.gpu>-1)
            sources = None

        cum_sentences += batch_size

        batch_size, src_len, hsize = encoding[0].size()

        with torch.no_grad():
            # for now
            if type(model) is Transformer:
                all_decodings = []
                decoding = model(encoding, source_masks, decoder_inputs, decoder_masks,
                                beam=args.beam_size, alpha=args.alpha, \
                                 decoding=True, feedback=attentions)
                all_decodings.append( decoding )
                num_iters = [0]

            elif type(model) is FastTransformer:
                decoding, all_decodings, num_iters, argmax_all_probs = run_fast_transformer(decoder_inputs.clone(), decoder_masks.clone(), \
                                            sources.clone(), source_masks.clone(), targets.clone(), [enc.clone() for enc in encoding], model, args, use_argmax=True)
                num_iters_total.extend( num_iters )

                if not args.use_argmax:
                    for _ in range(args.num_valid_samples):
                        _, _, _, sampled_all_probs = run_fast_transformer(decoder_inputs.clone(), decoder_masks.clone(), \
                                                    sources.clone(), source_masks.clone(), targets.clone(), [enc.clone() for enc in encoding], model, args, use_argmax=False)
                        for iter_ in range(args.valid_repeat_dec):
                            argmax_all_probs[iter_] = argmax_all_probs[iter_] + sampled_all_probs[iter_]

                    all_decodings = []
                    for iter_ in range(args.valid_repeat_dec):
                        argmax_all_probs[iter_] = argmax_all_probs[iter_] / (args.num_valid_samples + 1)
                        all_decodings.append(torch.max(argmax_all_probs[iter_], dim=-1)[-1])
                    decoding = all_decodings[-1]

        used_t = time.time() - start_t
        curr_time += used_t

        if args.dataset != "mscoco":
            if args.remove_repeats:
                outputs_unidx = [model.output_decoding(d) for d in [('src', sources), ('trg', targets), ('trg', remove_repeats_tensor(decoding))]]
            else:
                outputs_unidx = [model.output_decoding(d) for d in [('src', sources), ('trg', targets), ('trg', decoding)]]

        else:
            # make sure that 5 captions per each example
            num_captions = len(all_captions[0])
            for c in range(1, len(all_captions)):
                assert (num_captions == len(all_captions[c]))

            # untokenize reference captions
            for n_ref in range(len(all_captions)):
                n_caps = len(all_captions[0])
                for c in range(n_caps):
                    all_captions[n_ref][c] = all_captions[n_ref][c].replace("@@ ","")

            outputs_unidx = [ list(map(list, zip(*all_captions))) ]

        if args.remove_repeats:
            all_dec_outputs = [model.output_decoding(d) for d in [('trg', remove_repeats_tensor(all_decodings[ii])) for ii in range(len(all_decodings))]]
        else:
            all_dec_outputs = [model.output_decoding(d) for d in [('trg', all_decodings[ii]) for ii in range(len(all_decodings))]]

        corpus_size += batch_size
        if args.dataset != "mscoco":
            cum_tokens += sum([len(xx.split(" ")) for xx in outputs_unidx[0]]) # NOTE source tokens, not target

        if args.dataset != "mscoco":
            src_outputs += outputs_unidx[0]
            trg_outputs += outputs_unidx[1]
            if args.remove_repeats:
                dec_outputs += remove_repeats(outputs_unidx[-1])
            else:
                dec_outputs += outputs_unidx[-1]

        else:
            trg_outputs += outputs_unidx[0]

        for idx, each_output in enumerate(all_dec_outputs):
            if args.remove_repeats:
                all_decs[idx] += remove_repeats(each_output)
            else:
                all_decs[idx] += each_output

        #if True:
        if False and decoding_path is not None:
            for sent_i in range(len(outputs_unidx[0])):
                if args.dataset != "mscoco":
                    print ('SRC')
                    print (outputs_unidx[0][sent_i])
                    for ii in range(len(all_decodings)):
                        print ('DEC iter {}'.format(ii))
                        print (all_dec_outputs[ii][sent_i])
                    print ('TRG')
                    print (outputs_unidx[1][sent_i])
                else:
                    print ('TRG')
                    trg = outputs_unidx[0]
                    for subsent_i in range(len(trg[sent_i])):
                        print ('TRG {}'.format(subsent_i))
                        print (trg[sent_i][subsent_i])
                    for ii in range(len(all_decodings)):
                        print ('DEC iter {}'.format(ii))
                        print (all_dec_outputs[ii][sent_i])
                print ('---------------------------')

        timings += [used_t]

        if args.save_decs:
            for s, t, d in zip(outputs_unidx[0], outputs_unidx[1], outputs_unidx[2]):
                s, t, d = s.replace('@@ ', ''), t.replace('@@ ', ''), d.replace('@@ ', '')
                print(s, file=handles[0], flush=True)
                print(t, file=handles[1], flush=True)
                print(d, file=handles[2], flush=True)

        if not args.no_tqdm:
            progressbar.update(iters)
            progressbar.set_description('finishing sentences={}/batches={}, \
                length={}/average iter={}, speed={} sec/batch'.format(\
                corpus_size, iters, src_len, np.mean(np.array(num_iters)), curr_time / (1 + iters)))

    if evaluate:
        for idx, each_dec in enumerate(all_decs):
            if len(all_decs[idx]) != len(trg_outputs):
                break
            if args.dataset != "mscoco":
                bleu_output = computeBLEU(each_dec, trg_outputs, corpus=True, tokenizer=tokenizer)
            else:
                bleu_output = computeBLEUMSCOCO(each_dec, trg_outputs, corpus=True, tokenizer=tokenizer)
            args.logger.info("iter {} | {}".format(idx+1, print_bleu(bleu_output)))

    if args.adaptive_decoding != None:
        args.logger.info("----------------------------------------------")
        args.logger.info("Average # iters {}".format(np.mean(num_iters_total)))
        bleu_output = computeBLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
        args.logger.info("Adaptive BLEU | {}".format(print_bleu(bleu_output)))

    args.logger.info("----------------------------------------------")
    args.logger.info("Decoding speed analysis :")
    args.logger.info("{} sentences".format(cum_sentences))
    if args.dataset != "mscoco":
        args.logger.info("{} tokens".format(cum_tokens))
    args.logger.info("{:.3f} seconds".format(curr_time))

    args.logger.info("{:.3f} ms / sentence".format((curr_time / float(cum_sentences) * 1000)))
    if args.dataset != "mscoco":
        args.logger.info("{:.3f} ms / token".format((curr_time / float(cum_tokens) * 1000)))

    args.logger.info("{:.3f} sentences / s".format(float(cum_sentences) / curr_time))
    if args.dataset != "mscoco":
        args.logger.info("{:.3f} tokens / s".format(float(cum_tokens) / curr_time))
    args.logger.info("----------------------------------------------")

    if args.decode_which > 0:
        args.logger.info("Writing to special file")
        parent = decoding_path / "speed" / "b_{}{}".format(args.beam_size if args.model is Transformer else args.valid_repeat_dec,
                                                          "" if args.model is Transformer else "_{}".format(args.adaptive_decoding != None))
        args.logger.info(str(parent))
        parent.mkdir(parents=True, exist_ok=True)
        speed_handle = (parent / "results.{}".format(args.decode_which) ).open('w')

        print("----------------------------------------------", file=speed_handle, flush=True)
        print("Decoding speed analysis :", file=speed_handle, flush=True)
        print("{} sentences".format(cum_sentences), file=speed_handle, flush=True)
        if args.dataset != "mscoco":
            print("{} tokens".format(cum_tokens), file=speed_handle, flush=True)
        print("{:.3f} seconds".format(curr_time), file=speed_handle, flush=True)

        print("{:.3f} ms / sentence".format((curr_time / float(cum_sentences) * 1000)), file=speed_handle, flush=True)
        if args.dataset != "mscoco":
            print("{:.3f} ms / token".format((curr_time / float(cum_tokens) * 1000)), file=speed_handle, flush=True)

        print("{:.3f} sentences / s".format(float(cum_sentences) / curr_time), file=speed_handle, flush=True)
        if args.dataset != "mscoco":
            print("{:.3f} tokens / s".format(float(cum_tokens) / curr_time), file=speed_handle, flush=True)
        print("----------------------------------------------", file=speed_handle, flush=True)
