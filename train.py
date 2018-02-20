import ipdb
import torch
import numpy as np
import math
import gc
import os

import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from tqdm import tqdm, trange
from model import Transformer, FastTransformer, INF, TINY, softmax
from data import NormalField, NormalTranslationDataset, TripleTranslationDataset, ParallelDataset, data_path
from utils import Metrics, Best, TargetLength, computeBLEU, computeBLEUMSCOCO, compute_bp, Batch, masked_sort, computeGroupBLEU, \
        corrupt_target, remove_repeats, remove_repeats_tensor, print_bleu, corrupt_target_fix, set_eos, organise_trg_len_dic
from time import gmtime, strftime

# helper functions
def export(x):
    try:
        with torch.cuda.device_of(x):
            return x.data.cpu().float().mean()
    except Exception:
        return 0

tokenizer = lambda x: x.replace('@@ ', '').split()

def valid_model(args, model, dev, dev_metrics=None, dev_metrics_trg=None, dev_metrics_average=None,
                print_out=False, teacher_model=None, trg_len_dic=None):
    print_seq = (['REF '] if args.dataset == "mscoco" else ['SRC ', 'REF ']) + ['HYP{}'.format(ii+1) for ii in range(args.valid_repeat_dec)]

    trg_outputs = []
    real_all_outputs = [ [] for ii in range(args.valid_repeat_dec)]
    short_all_outputs = [ [] for ii in range(args.valid_repeat_dec)]
    outputs_data = {}

    model.eval()
    for j, dev_batch in enumerate(dev):
        if args.dataset == "mscoco":
            # only use first caption for calculating log likelihood
            all_captions = dev_batch[1]
            dev_batch[1] = dev_batch[1][0]
            decoder_inputs, decoder_masks,\
            targets, target_masks,\
            _, source_masks,\
            encoding, batch_size, rest = model.quick_prepare_mscoco(dev_batch, all_captions=all_captions, fast=(type(model) is FastTransformer), inputs_dec=args.inputs_dec, trg_len_option=args.trg_len_option, max_len=args.max_offset, trg_len_dic=trg_len_dic, bp=args.bp)

        else:
            decoder_inputs, decoder_masks,\
            targets, target_masks,\
            sources, source_masks,\
            encoding, batch_size, rest = model.quick_prepare(dev_batch, fast=(type(model) is FastTransformer), trg_len_option=args.trg_len_option, trg_len_ratio=args.trg_len_ratio, trg_len_dic=trg_len_dic, bp=args.bp)

        losses, all_decodings = [], []
        if type(model) is Transformer:
            decoding, out, probs = model(encoding, source_masks, decoder_inputs, decoder_masks, beam=1, decoding=True, return_probs=True)
            loss = model.cost(targets, target_masks, out=out)
            losses.append(loss)
            all_decodings.append( decoding )

        elif type(model) is FastTransformer:
            for iter_ in range(args.valid_repeat_dec):
                curr_iter = min(iter_, args.num_decs-1)
                next_iter = min(curr_iter + 1, args.num_decs-1)

                decoding, out, probs = model(encoding, source_masks, decoder_inputs, decoder_masks, decoding=True, return_probs=True, iter_=curr_iter)

                loss = model.cost(targets, target_masks, out=out, iter_=curr_iter)
                losses.append(loss)
                all_decodings.append( decoding )

                decoder_inputs = 0
                if args.next_dec_input in ["both", "emb"]:
                    _, argmax = torch.max(probs, dim=-1)
                    emb = F.embedding(argmax, model.decoder[next_iter].out.weight * math.sqrt(args.d_model))
                    decoder_inputs += emb

                if args.next_dec_input in ["both", "out"]:
                    decoder_inputs += out

        if args.dataset == "mscoco":
            # make sure that 5 captions per each example
            num_captions = len(all_captions[0])
            for c in range(1, len(all_captions)):
                assert (num_captions == len(all_captions[c]))

            # untokenize reference captions
            for n_ref in range(len(all_captions)):
                n_caps = len(all_captions[0])
                for c in range(n_caps):
                    all_captions[n_ref][c] = all_captions[n_ref][c].replace("@@ ","")

            src_ref = [ list(map(list, zip(*all_captions))) ]
        else:
            src_ref = [ model.output_decoding(d) for d in [('src', sources), ('trg', targets)] ]

        real_outputs = [ model.output_decoding(d) for d in [('trg', xx) for xx in all_decodings] ]

        if print_out:
            if args.dataset != "mscoco":
                for k, d in enumerate(src_ref + real_outputs):
                    args.logger.info("{} ({}): {}".format(print_seq[k], len(d[0].split(" ")), d[0]))
            else:
                for k in range(len(all_captions[0])):
                    for c in range(len(all_captions)):
                        args.logger.info("REF ({}): {}".format(len(all_captions[c][k].split(" ")), all_captions[c][k]))

                    for c in range(len(real_outputs)):
                        args.logger.info("HYP {} ({}): {}".format(c+1, len(real_outputs[c][k].split(" ")),  real_outputs[c][k]))
            args.logger.info('------------------------------------------------------------------')

        trg_outputs += src_ref[-1]
        for ii, d_outputs in enumerate(real_outputs):
            real_all_outputs[ii] += d_outputs

        if dev_metrics is not None:
            dev_metrics.accumulate(batch_size, *losses)
        if dev_metrics_trg is not None:
            dev_metrics_trg.accumulate(batch_size, *[rest[0], rest[1], rest[2]])
        if dev_metrics_average is not None:
            dev_metrics_average.accumulate(batch_size, *[rest[3], rest[4]])

    if args.dataset != "mscoco":
        real_bleu = [computeBLEU(ith_output, trg_outputs, corpus=True, tokenizer=tokenizer) for ith_output in real_all_outputs]
    else:
        real_bleu = [computeBLEUMSCOCO(ith_output, trg_outputs, corpus=True, tokenizer=tokenizer) for ith_output in real_all_outputs]

    outputs_data['real'] = real_bleu

    if "predict" in args.trg_len_option:
        outputs_data['pred_target_len_loss'] = getattr(dev_metrics_trg, 'pred_target_len_loss')
        outputs_data['pred_target_len_correct'] = getattr(dev_metrics_trg, 'pred_target_len_correct')
        outputs_data['pred_target_len_approx'] = getattr(dev_metrics_trg, 'pred_target_len_approx')
        outputs_data['average_target_len_correct'] = getattr(dev_metrics_average, 'average_target_len_correct')
        outputs_data['average_target_len_approx'] = getattr(dev_metrics_average, 'average_target_len_approx')

    if dev_metrics is not None:
        args.logger.info(dev_metrics)
    if dev_metrics_trg is not None:
        args.logger.info(dev_metrics_trg)
    if dev_metrics_average is not None:
        args.logger.info(dev_metrics_average)

    for idx in range(args.valid_repeat_dec):
        print_str = "iter {} | {}".format(idx+1, print_bleu(real_bleu[idx], verbose=False))
        args.logger.info( print_str )

    return outputs_data

def train_model(args, model, train, dev, src=None, trg=None, trg_len_dic=None, teacher_model=None, save_path=None, maxsteps=None):

    if args.tensorboard and (not args.debug):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(str(args.event_path / args.id_str))

    if type(model) is FastTransformer and args.denoising_prob > 0.0:
        denoising_weights = [args.denoising_weight for idx in range(args.train_repeat_dec)]
        denoising_out_weights = [args.denoising_out_weight for idx in range(args.train_repeat_dec)]

    if type(model) is FastTransformer and args.layerwise_denoising_weight:
        start, end = 0.9, 0.1
        diff = (start-end)/(args.train_repeat_dec-1)
        denoising_weights = np.arange(start=end, stop=start, step=diff).tolist()[::-1] + [0.1]

    # optimizer
    for k, p in zip(model.state_dict().keys(), model.parameters()):
        # only finetune layers that are responsible to predicting target len
        if args.finetune_trg_len:
            if "pred_len" not in k:
                p.requires_grad = False
        else:
            if "pred_len" in k:
                p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9)
    else:
        raise NotImplementedError

    # if resume training
    if (args.load_from is not None) and (args.resume):
        with torch.cuda.device(args.gpu):   # very important.
            offset, opt_states = torch.load(str(args.model_path / args.load_from) + '.pt.states',
                                            map_location=lambda storage, loc: storage.cuda())
            opt.load_state_dict(opt_states)
    else:
        offset = 0

    if not args.finetune_trg_len:
        best = Best(max, *['BLEU_dec{}'.format(ii+1) for ii in range(args.valid_repeat_dec)],
                         'i', model=model, opt=opt, path=str(args.model_path / args.id_str), gpu=args.gpu,
                         which=range(args.valid_repeat_dec))
    else:
        best = Best(max, *['pred_target_len_correct'],
                         'i', model=model, opt=opt, path=str(args.model_path / args.id_str), gpu=args.gpu,
                         which=[0])
    train_metrics = Metrics('train loss', *['loss_{}'.format(idx+1) for idx in range(args.train_repeat_dec)], data_type = "avg")
    dev_metrics = Metrics('dev loss', *['loss_{}'.format(idx+1) for idx in range(args.valid_repeat_dec)], data_type = "avg")

    if "predict" in args.trg_len_option:
        train_metrics_trg = Metrics('train loss target', *["pred_target_len_loss", "pred_target_len_correct", "pred_target_len_approx"], data_type="avg")
        train_metrics_average = Metrics('train loss average', *["average_target_len_correct", "average_target_len_approx"], data_type="avg")
        dev_metrics_trg = Metrics('dev loss target', *["pred_target_len_loss", "pred_target_len_correct", "pred_target_len_approx"], data_type="avg")
        dev_metrics_average = Metrics('dev loss average', *["average_target_len_correct", "average_target_len_approx"], data_type="avg")
    else:
        train_metrics_trg = None
        train_metrics_average = None
        dev_metrics_trg = None
        dev_metrics_average = None

    if not args.no_tqdm:
        progressbar = tqdm(total=args.eval_every, desc='start training.')

    if maxsteps is None:
        maxsteps = args.maximum_steps

    #targetlength = TargetLength()
    for iters, train_batch in enumerate(train):
        #targetlength.accumulate( train_batch )
        #continue

        iters += offset

        if args.save_every > 0 and iters % args.save_every == 0:
            args.logger.info('save (back-up) checkpoints at iter={}'.format(iters))
            with torch.cuda.device(args.gpu):
                torch.save(best.model.state_dict(), '{}_iter={}.pt'.format(str(args.model_path / args.id_str), iters))
                torch.save([iters, best.opt.state_dict()], '{}_iter={}.pt.states'.format(str(args.model_path / args.id_str), iters))

        if iters % args.eval_every == 0:
            torch.cuda.empty_cache()
            gc.collect()
            dev_metrics.reset()
            if dev_metrics_trg is not None:
                dev_metrics_trg.reset()
            if dev_metrics_average is not None:
                dev_metrics_average.reset()
            outputs_data = valid_model(args, model, dev, dev_metrics, dev_metrics_trg=dev_metrics_trg, dev_metrics_average=dev_metrics_average, teacher_model=None, print_out=True, trg_len_dic=trg_len_dic)
            #outputs_data = [0, [0,0,0,0], 0, 0]
            if args.tensorboard and (not args.debug):
                for ii in range(args.valid_repeat_dec):
                    writer.add_scalar('dev/single/Loss_{}'.format(ii + 1), getattr(dev_metrics, "loss_{}".format(ii+1)), iters) # NLL averaged over dev corpus
                    writer.add_scalar('dev/single/BLEU_{}'.format(ii + 1), outputs_data['real'][ii][0], iters) # NOTE corpus bleu

                if "predict" in args.trg_len_option:
                    writer.add_scalar("dev/single/pred_target_len_loss", outputs_data["pred_target_len_loss"], iters)
                    writer.add_scalar("dev/single/pred_target_len_correct", outputs_data["pred_target_len_correct"], iters)
                    writer.add_scalar("dev/single/pred_target_len_approx", outputs_data["pred_target_len_approx"], iters)
                    writer.add_scalar("dev/single/average_target_len_correct", outputs_data["average_target_len_correct"], iters)
                    writer.add_scalar("dev/single/average_target_len_approx", outputs_data["average_target_len_approx"], iters)

                """
                writer.add_scalars('dev/total/BLEUs', {"iter_{}".format(idx+1):bleu for idx, bleu in enumerate(outputs_data['bleu']) }, iters)
                writer.add_scalars('dev/total/Losses',
                    { "iter_{}".format(idx+1):getattr(dev_metrics, "loss_{}".format(idx+1))
                     for idx in range(args.valid_repeat_dec) },
                     iters )
                """

            if not args.debug:
                if not args.finetune_trg_len:
                    best.accumulate(*[xx[0] for xx in outputs_data['real']], iters)

                    values = list( best.metrics.values() )
                    args.logger.info("best model : {}, {}".format( "BLEU=[{}]".format(", ".join( [ str(x) for x in values[:args.valid_repeat_dec] ] ) ), \
                                                                  "i={}".format( values[args.valid_repeat_dec] ), ) )
                else:
                    best.accumulate(*[outputs_data['pred_target_len_correct']], iters)
                    values = list( best.metrics.values() )
                    args.logger.info("best model : {}".format( "pred_target_len_correct = {}".format(values[0])) )

            args.logger.info('model:' + args.prefix + args.hp_str)

            # ---set-up a new progressor---
            if not args.no_tqdm:
                progressbar.close()
                progressbar = tqdm(total=args.eval_every, desc='start training.')

            if type(model) is FastTransformer and args.anneal_denoising_weight:
                for ii, bb in enumerate([xx[0] for xx in outputs_data['real']][:-1]):
                    denoising_weights[ii] = 0.9 - 0.1 * int(math.floor(bb / 3.0))

        if iters > maxsteps:
            args.logger.info('reached the maximum updating steps.')
            break

        model.train()

        def get_lr_transformer(i, lr0=0.1):
            return lr0 * 10 / math.sqrt(args.d_model) * min(
                    1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))

        def get_lr_anneal(iters, lr0=0.1):
            lr_end = 1e-5
            return max( 0, (args.lr - lr_end) * (args.anneal_steps - iters) / args.anneal_steps ) + lr_end

        if args.lr_schedule == "fixed":
            opt.param_groups[0]['lr'] = args.lr
        elif args.lr_schedule == "anneal":
            opt.param_groups[0]['lr'] = get_lr_anneal(iters + 1)
        elif args.lr_schedule == "transformer":
            opt.param_groups[0]['lr'] = get_lr_transformer(iters + 1)

        opt.zero_grad()

        if args.dataset == "mscoco":
            decoder_inputs, decoder_masks,\
            targets, target_masks,\
            _, source_masks,\
            encoding, batch_size, rest = model.quick_prepare_mscoco(train_batch, all_captions=train_batch[1], fast=(type(model) is FastTransformer), inputs_dec=args.inputs_dec, trg_len_option=args.trg_len_option, max_len=args.max_offset, trg_len_dic=trg_len_dic, bp=args.bp)
        else:
            decoder_inputs, decoder_masks,\
            targets, target_masks,\
            sources, source_masks,\
            encoding, batch_size, rest = model.quick_prepare(train_batch, fast=(type(model) is FastTransformer), trg_len_option=args.trg_len_option, trg_len_ratio=args.trg_len_ratio, trg_len_dic=trg_len_dic, bp=args.bp)

        losses = []
        if type(model) is Transformer:
            loss = model.cost(targets, target_masks, out=model(encoding, source_masks, decoder_inputs, decoder_masks))
            losses.append( loss )

        elif type(model) is FastTransformer:
            all_logits = []
            all_denoising_masks = []
            for iter_ in range(args.train_repeat_dec):
                curr_iter = min(iter_, args.num_decs-1)
                next_iter = min(curr_iter + 1, args.num_decs-1)

                out = model(encoding, source_masks, decoder_inputs, decoder_masks, iter_=curr_iter, return_probs=False)

                if args.self_distil > 0.0:
                    loss, logits_masked = model.cost(targets, target_masks, out=out, iter_=curr_iter, return_logits=True)
                else:
                    loss = model.cost(targets, target_masks, out=out, iter_=curr_iter)

                logits = model.decoder[curr_iter].out(out)

                if args.use_argmax:
                    _, argmax = torch.max(logits, dim=-1)
                else:
                    probs = softmax(logits)
                    probs_sz = probs.size()
                    logits_ = Variable(probs.data, requires_grad=False)
                    argmax = torch.multinomial(logits_.contiguous().view(-1, probs_sz[-1]), 1).view(*probs_sz[:-1])

                if args.self_distil > 0.0:
                    all_logits.append(logits_masked)

                losses.append(loss)

                decoder_inputs_ = 0
                denoising_mask = 1
                if args.next_dec_input in ["both", "emb"]:
                    if args.denoising_prob > 0.0 and np.random.rand() < args.denoising_prob:
                        cor = corrupt_target(targets, decoder_masks, len(trg.vocab), denoising_weights[iter_], args.corruption_probs)

                        emb = F.embedding(cor, model.decoder[next_iter].out.weight * math.sqrt(args.d_model))
                        denoising_mask = 0
                    else:
                        emb = F.embedding(argmax, model.decoder[next_iter].out.weight * math.sqrt(args.d_model))

                    if args.denoising_out_weight > 0:
                        if denoising_out_weights[iter_] > 0.0:
                            corrupted_argmax = corrupt_target(argmax, decoder_masks, denoising_out_weights[iter_])
                        else:
                            corrupted_argmax = argmax
                        emb = F.embedding(corrupted_argmax, model.decoder[next_iter].out.weight * math.sqrt(args.d_model))
                    decoder_inputs_ += emb
                all_denoising_masks.append( denoising_mask )

                if args.next_dec_input in ["both", "out"]:
                    decoder_inputs_ += out
                decoder_inputs = decoder_inputs_

            # self distillation loss if requested
            if args.self_distil > 0.0:
                self_distil_losses = []

                for logits_i in range(1, len(all_logits)-1):
                    self_distill_loss_i = 0.0
                    for logits_j in range(logits_i+1, len(all_logits)):
                        self_distill_loss_i += \
                                all_denoising_masks[logits_j] * \
                                all_denoising_masks[logits_i] * \
                                (1/(logits_j-logits_i)) * args.self_distil * F.mse_loss(all_logits[logits_i], all_logits[logits_j].detach())

                    self_distil_losses.append(self_distill_loss_i)

                self_distil_loss = sum(self_distil_losses)

        loss = sum(losses)

        # accmulate the training metrics
        train_metrics.accumulate(batch_size, *losses, print_iter=None)
        if train_metrics_trg is not None:
            train_metrics_trg.accumulate(batch_size, *[rest[0], rest[1], rest[2]])
        if train_metrics_average is not None:
            train_metrics_average.accumulate(batch_size, *[rest[3], rest[4]])
        if type(model) is FastTransformer and args.self_distil > 0.0:
            (loss+self_distil_loss).backward()
        else:
            if "predict" in args.trg_len_option:
                if args.finetune_trg_len:
                    rest[0].backward()
                else:
                    loss.backward()
            else:
                loss.backward()

        if args.grad_clip > 0:
            total_norm = nn.utils.clip_grad_norm(params, args.grad_clip)
        opt.step()

        mid_str = ''
        if type(model) is FastTransformer and args.self_distil > 0.0:
            mid_str += 'distil={:.5f}, '.format(self_distil_loss.cpu().data.numpy()[0])
        if type(model) is FastTransformer and "predict" in args.trg_len_option:
            mid_str += 'pred_target_len_loss={:.5f}, '.format(rest[0].cpu().data.numpy()[0])
        if type(model) is FastTransformer and args.denoising_prob > 0.0:
            mid_str += "/".join(["{:.1f}".format(ff) for ff in denoising_weights[:-1]])+", "

        info = 'update={}, loss={}, {}lr={:.1e}'.format( iters,
                    "/".join(["{:.3f}".format(export(ll)) for ll in losses]),
                    mid_str,
                    opt.param_groups[0]['lr'])

        if args.no_tqdm:
            if iters % args.eval_every == 0:
                args.logger.info("update {} : {}".format(iters, str(train_metrics)))
        else:
            progressbar.update(1)
            progressbar.set_description(info)

        if iters % args.eval_every == 0 and args.tensorboard and (not args.debug):
            for idx in range(args.train_repeat_dec):
                writer.add_scalar('train/single/Loss_{}'.format(idx+1), getattr(train_metrics, "loss_{}".format(idx+1)), iters)
            if "predict" in args.trg_len_option:
                writer.add_scalar("train/single/pred_target_len_loss", getattr(train_metrics_trg, "pred_target_len_loss"), iters)
                writer.add_scalar("train/single/pred_target_len_correct", getattr(train_metrics_trg, "pred_target_len_correct"), iters)
                writer.add_scalar("train/single/pred_target_len_approx", getattr(train_metrics_trg, "pred_target_len_approx"), iters)
                writer.add_scalar("train/single/average_target_len_correct", getattr(train_metrics_average, "average_target_len_correct"), iters)
                writer.add_scalar("train/single/average_target_len_approx", getattr(train_metrics_average, "average_target_len_approx"), iters)

            train_metrics.reset()
            if train_metrics_trg is not None:
                train_metrics_trg.reset()
            if train_metrics_average is not None:
                train_metrics_average.reset()

    #torch.save(targetlength.lengths, str(args.data_prefix / "trg_len_dic" / args.dataset[-4:]))
