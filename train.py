import torch
import numpy as np
import math
import gc
import os

import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from tqdm import tqdm, trange
from model import Transformer, MultiGPUTransformer, MultiGPUFastTransformer, FastTransformer, INF, TINY, softmax
from data import NormalField, NormalTranslationDataset, TripleTranslationDataset, ParallelDataset, data_path
from utils import Metrics, Best, TargetLength, computeBLEU, computeBLEUMSCOCO, compute_bp, Batch, masked_sort, computeGroupBLEU, \
        corrupt_target, remove_repeats, remove_repeats_tensor, print_bleu, corrupt_target_fix, set_eos, organise_trg_len_dic, evaluate_pred_target_len
from time import gmtime, strftime

from model_utils import prepare_sources, prepare_decoder_inputs, prepare_targets, linear_attention, apply_mask

import glob
import time

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
    if args.trg_len_option == "predict":
        pred_all_outputs = [ [] for ii in range(args.valid_repeat_dec)]
    outputs_data = {}

    model.eval()
    for j, dev_batch in enumerate(dev):

        # use only one gpu to do inference
        if type(model) is nn.DataParallel:
            inference_model = model.module.model
        else:
            inference_model = model

        # process dev batch first
        if args.trg_len_option == "predict":
            decoder_inputs, decoder_masks, pred_decoder_inputs, pred_decoder_masks,\
            targets, target_masks,\
            sources, source_masks,\
            encoding, batch_size, rest = inference_model.quick_prepare(dev_batch, fast=(type(inference_model) is FastTransformer), trg_len_option=args.trg_len_option, trg_len_ratio=args.trg_len_ratio, trg_len_dic=trg_len_dic, bp=args.bp)
        else:
            decoder_inputs, decoder_masks,\
            targets, target_masks,\
            sources, source_masks,\
            encoding, batch_size, rest = inference_model.quick_prepare(dev_batch, fast=(type(inference_model) is FastTransformer), trg_len_option=args.trg_len_option, trg_len_ratio=args.trg_len_ratio, trg_len_dic=trg_len_dic, bp=args.bp)

        losses, all_decodings = [], []
        if args.trg_len_option == "predict":
            pred_all_decodings = []

        with torch.no_grad():
            if type(inference_model) is Transformer:
                decoding, out, probs = inference_model(encoding, source_masks, decoder_inputs, decoder_masks, beam=1, decoding=True, return_probs=True)
                loss = inference_model.cost(targets, target_masks, out=out)
                losses.append(loss)
                all_decodings.append( decoding )

            elif type(inference_model) is FastTransformer:
                for iter_ in range(args.valid_repeat_dec):
                    curr_iter = min(iter_, args.num_decs-1)
                    next_iter = min(curr_iter + 1, args.num_decs-1)

                    decoding, out, probs = inference_model(encoding, source_masks, decoder_inputs, decoder_masks, decoding=True, return_probs=True, iter_=curr_iter)

                    loss = inference_model.cost(targets, target_masks, out=out, iter_=curr_iter)
                    losses.append(loss)
                    all_decodings.append( decoding )

                    decoder_inputs = 0
                    if args.next_dec_input in ["both", "emb"]:
                        _, argmax = torch.max(probs, dim=-1)
                        emb = F.embedding(argmax, inference_model.decoder[next_iter].out.weight * math.sqrt(args.d_model))
                        decoder_inputs += emb

                    if args.next_dec_input in ["both", "out"]:
                        decoder_inputs += out

                    # if trg_len_option is predict
                    if args.trg_len_option == "predict":
                        pred_decoding, pred_out, pred_probs = inference_model(encoding, source_masks, pred_decoder_inputs, pred_decoder_masks, decoding=True, return_probs=True, iter_=curr_iter)
                        pred_all_decodings.append(pred_decoding)

                        pred_decoder_inputs = 0
                        if args.next_dec_input in ["both", "emb"]:
                            _, pred_argmax = torch.max(pred_probs, dim=-1)
                            pred_emb = F.embedding(pred_argmax, inference_model.decoder[next_iter].out.weight * math.sqrt(args.d_model))
                            pred_decoder_inputs += pred_emb

                        if args.next_dec_input in ["both", "out"]:
                            pred_decoder_inputs += pred_out

        src_ref = [ inference_model.output_decoding(d) for d in [('src', sources), ('trg', targets)] ]
        real_outputs = [ inference_model.output_decoding(d) for d in [('trg', xx) for xx in all_decodings] ]
        if args.trg_len_option == "predict":
            pred_outputs = [ inference_model.output_decoding(d) for d in [('trg', xx) for xx in pred_all_decodings] ]

        if print_out:
            for k, d in enumerate(src_ref + real_outputs):
                args.logger.info("{} ({}): {}".format(print_seq[k], len(d[0].split(" ")), d[0]))
            args.logger.info('------------------------------------------------------------------')

        trg_outputs += src_ref[-1]
        if args.trg_len_option == "predict":
            for ii, (d_outputs, d_outputs_pred) in enumerate(zip(real_outputs, pred_outputs)):
                real_all_outputs[ii] += d_outputs
                pred_all_outputs[ii] += d_outputs_pred
        else:
            for ii, d_outputs in enumerate(real_outputs):
                real_all_outputs[ii] += d_outputs

        if dev_metrics is not None:
            dev_metrics.accumulate(batch_size, *losses)
        if dev_metrics_trg is not None:
            dev_metrics_trg.accumulate(batch_size, *[rest[0], rest[1], rest[2]])
        if dev_metrics_average is not None:
            dev_metrics_average.accumulate(batch_size, *[rest[3], rest[4]])

    real_bleu = [computeBLEU(ith_output, trg_outputs, corpus=True, tokenizer=tokenizer) for ith_output in real_all_outputs]
    outputs_data['real'] = real_bleu

    if "predict" in args.trg_len_option:
        # for statistics
        pred_bleu = [computeBLEU(ith_output, trg_outputs, corpus=True, tokenizer=tokenizer) for ith_output in pred_all_outputs]
        outputs_data['pred'] = pred_bleu

        outputs_data['pred_target_len_loss'] = getattr(dev_metrics_trg, 'pred_target_len_loss')
        outputs_data['pred_target_len_correct'] = getattr(dev_metrics_trg, 'pred_target_len_correct')
        outputs_data['pred_target_len_approx'] = getattr(dev_metrics_trg, 'pred_target_len_approx')
        if dev_metrics_average is not None:
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

    if args.trg_len_option == "predict":
        for idx in range(args.valid_repeat_dec):
            print_str = "(pred) iter {} | {}".format(idx+1, print_bleu(pred_bleu[idx], verbose=False))
            args.logger.info( print_str )

    return outputs_data

def collect_garbage():
    torch.cuda.empty_cache()
    gc.collect()

def train_model(args, model, train, dev, src=None, trg=None, trg_len_dic=None, teacher_model=None, save_path=None, maxsteps=None):

    if args.tensorboard and (not args.debug):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(str(args.event_path / args.id_str))

    if type(model.module) is MultiGPUFastTransformer and args.denoising_prob > 0.0:
        denoising_weights = [args.denoising_weight for idx in range(args.train_repeat_dec)]
        denoising_out_weights = [args.denoising_out_weight for idx in range(args.train_repeat_dec)]

    if type(model.module) is MultiGPUFastTransformer and args.layerwise_denoising_weight:
        start, end = 0.9, 0.1
        diff = (start-end)/(args.train_repeat_dec-1)
        denoising_weights = np.arange(start=end, stop=start, step=diff).tolist()[::-1] + [0.1]

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

    best = Best(max, *['BLEU_dec{}'.format(ii+1) for ii in range(args.valid_repeat_dec)],
                     'i', model=model, opt=opt, path=str(args.model_path / args.id_str), gpu=args.gpu, num_gpus=args.num_gpus,
                     which=range(args.valid_repeat_dec))

    if args.trg_len_option == "predict":
        best_pred = Best(max, *['BLEU_dec_pred{}'.format(ii+1) for ii in range(args.valid_repeat_dec)],
                         'i', model=model, opt=opt, path=str(args.model_path / args.id_str)+"pred_", gpu=args.gpu, num_gpus=args.num_gpus,
                         which=range(args.valid_repeat_dec))

    train_metrics = Metrics('train loss', *['loss_{}'.format(idx+1) for idx in range(args.train_repeat_dec)], data_type = "avg")
    dev_metrics = Metrics('dev loss', *['loss_{}'.format(idx+1) for idx in range(args.valid_repeat_dec)], data_type = "avg")

    # train and dev metrics for tracking accuracy of predicting target len
    if args.trg_len_option == "predict":
        train_metrics_trg = Metrics('train loss target', *["pred_target_len_loss", "pred_target_len_correct", "pred_target_len_approx"], data_type="avg")
        dev_metrics_trg = Metrics('dev loss target', *["pred_target_len_loss", "pred_target_len_correct", "pred_target_len_approx"], data_type="avg")
        if trg_len_dic != None:
            train_metrics_average = Metrics('train loss average', *["average_target_len_correct", "average_target_len_approx"], data_type="avg")
            dev_metrics_average = Metrics('dev loss average', *["average_target_len_correct", "average_target_len_approx"], data_type="avg")
        else:
            train_metrics_average = None
            dev_metrics_average = None
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

        if args.save_every > 0 and iters % args.save_every == 0 and (not args.debug):
            args.logger.info('save (back-up) checkpoints at iter={}'.format(iters))
            if args.save_last:
                # if save_last then delete the checkpoint before
                prev_files = glob.glob('{}_iter=*'.format(str(args.model_path / args.id_str)))
                for prev_f in prev_files:
                    os.remove(prev_f)
            with torch.cuda.device(args.gpu):
                if args.num_gpus > 1:
                    torch.save(best.model.module.model.state_dict(), '{}_iter={}.pt'.format(str(args.model_path / args.id_str), iters))
                else:
                    torch.save(best.model.state_dict(), '{}_iter={}.pt'.format(str(args.model_path / args.id_str), iters))
                torch.save([iters, best.opt.state_dict()], '{}_iter={}.pt.states'.format(str(args.model_path / args.id_str), iters))

        if iters % args.eval_every == 0:
            collect_garbage()

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
                    if args.trg_len_option == "predict":
                        writer.add_scalar('dev/single/pred_BLEU_{}'.format(ii + 1), outputs_data['pred'][ii][0], iters) # NOTE corpus bleu

                if "predict" in args.trg_len_option:
                    writer.add_scalar("dev/single/pred_target_len_loss", outputs_data["pred_target_len_loss"], iters)
                    writer.add_scalar("dev/single/pred_target_len_correct", outputs_data["pred_target_len_correct"], iters)
                    writer.add_scalar("dev/single/pred_target_len_approx", outputs_data["pred_target_len_approx"], iters)
                    if trg_len_dic is not None:
                        writer.add_scalar("dev/single/average_target_len_correct", outputs_data["average_target_len_correct"], iters)
                        writer.add_scalar("dev/single/average_target_len_approx", outputs_data["average_target_len_approx"], iters)

            if not args.debug:
                best.accumulate(*[xx[0] for xx in outputs_data['real']], iters)
                values = list( best.metrics.values() )
                args.logger.info("best model : {}, {}".format( "BLEU=[{}]".format(", ".join( [ str(x) for x in values[:args.valid_repeat_dec] ] ) ), \
                                                              "i={}".format( values[args.valid_repeat_dec] ), ) )
                if args.trg_len_option == "predict":
                    best_pred.accumulate(*[xx[0] for xx in outputs_data['pred']], iters)
                    values_pred = list( best_pred.metrics.values() )
                    args.logger.info("(pred) best model : {}, {}".format( "BLEU=[{}]".format(", ".join( [ str(x) for x in values_pred[:args.valid_repeat_dec] ] ) ), \
                                                                  "i={}".format( values_pred[args.valid_repeat_dec] ), ) )

            args.logger.info('model:' + args.prefix + args.hp_str)

            # ---set-up a new progressor---
            if not args.no_tqdm:
                progressbar.close()
                progressbar = tqdm(total=args.eval_every, desc='start training.')

            if type(model.module) is MultiGPUFastTransformer and args.anneal_denoising_weight:
                for ii, bb in enumerate([xx[0] for xx in outputs_data['real']][:-1]):
                    denoising_weights[ii] = 0.9 - 0.1 * int(math.floor(bb / 3.0))

            collect_garbage()

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

        # make sure
        assert (train_batch.batch_size % args.num_gpus == 0)

        losses = [0 for _ in range(args.train_repeat_dec)]

        if type(model) is nn.DataParallel:
            if type(model.module) == MultiGPUTransformer:
                # prepare_sources, prepare_decoder_inputs, prepare_targets
                sources, source_masks = prepare_sources(train_batch, field=trg)
                targets, target_masks = prepare_targets(train_batch, field=trg)
                decoder_inputs, decoder_masks = prepare_decoder_inputs(train_batch.trg, field=trg)
                loss = model(sources.cuda(), source_masks.cuda(), decoder_inputs.cuda(), decoder_masks.cuda(), targets.cuda(), target_masks.cuda())
                loss.backward(torch.ones_like(loss.data))
                losses[0] = loss.mean()

            elif type(model.module) == MultiGPUFastTransformer:
                # first decide when to denoise
                denoise = [False] * args.train_repeat_dec
                if args.denoising_prob > 0.0:
                    for iter_ in range(args.train_repeat_dec):
                        denoise[iter_] = args.denoising_prob > 0.0 and np.random.rand() < args.denoising_prob

                # prepare input data (prepare_sources, etc)
                sources, source_masks = prepare_sources(train_batch, field=trg)
                targets, target_masks = prepare_targets(train_batch, field=trg)
                _, decoder_masks = prepare_decoder_inputs(train_batch.trg, field=trg)

                # then prepare decoder inputs
                attention = linear_attention(source_masks, decoder_masks, args.decoder_input_how)
                attention = apply_mask(attention, decoder_masks, p=1)
                attention = attention[:,:,None].expand(*attention.size(), args.d_model)

                ### MULTI SAMPLE CASE
                for sub_iters in range(args.num_train_samples):
                    # prepare corrupted targets
                    denoise_cor = [None] * args.train_repeat_dec
                    if args.denoising_prob > 0.0:
                        for iter_ in range(args.train_repeat_dec):
                            cor = corrupt_target(targets, decoder_masks, len(trg.vocab), denoising_weights[iter_], args.corruption_probs)
                            denoise_cor[iter_] = cor.cuda()

                    # forward model get losses
                    # loss is now a tensor of batch_size x num_tokens
                    if args.trg_len_option == "predict":
                        loss, pred_target_len_loss, pred_target_len = \
                            model(sources.clone(), source_masks.clone(), attention.clone(),\
                                decoder_masks.clone(), targets.clone(), target_masks.clone(), \
                                denoise, denoise_cor, trg_len_option=args.trg_len_option)
                    else:
                        loss = \
                            model(sources.clone(), source_masks.clone(), attention.clone(),\
                                decoder_masks.clone(), targets.clone(), target_masks.clone(), \
                                denoise, denoise_cor, trg_len_option=args.trg_len_option)

                    # calculate loss
                    for train_repeat_iter in range(args.train_repeat_dec):
                        ### NO NEED TO DIVIDE BY (1/args.num_train_samples) BECAUSE ADAM IS USED
                        loss[train_repeat_iter] = loss[train_repeat_iter].mean()
                        # for statistics
                        losses[train_repeat_iter] += (1/args.num_train_samples) * loss[train_repeat_iter].clone().detach()

                    loss = sum(loss)
                    if args.trg_len_option == "predict":
                        pred_target_len_loss = pred_target_len_loss.mean()
                        loss = loss + pred_target_len_loss

                    # do backward pass
                    loss.backward(torch.ones_like(loss.data))

                if args.trg_len_option == "predict":
                    pred_target_len_results = evaluate_pred_target_len(source_masks, target_masks, pred_target_len, trg_len_dic)

        # accmulate the training metrics
        train_metrics.accumulate(train_batch.batch_size*args.num_gpus, *losses, print_iter=None)
        if args.trg_len_option == "predict":
            train_metrics_trg.accumulate(train_batch.batch_size*args.num_gpus, *[pred_target_len_loss, pred_target_len_results[0], pred_target_len_results[1]])
            if trg_len_dic != None:
                train_metrics_average.accumulate(train_batch.batch_size*args.num_gpus, *[pred_target_len_results[2], pred_target_len_results[3]])

        #loss.backward(torch.ones_like(loss.data))
        #t2_model = time.time()

        if args.grad_clip > 0:
            total_norm = nn.utils.clip_grad_norm(params, args.grad_clip)
        else:
            total_norm, norm_type = 0, 2
            for param in params:
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm ** norm_type
            total_norm = total_norm ** (1. / norm_type)

        if args.tensorboard and (not args.debug):
            writer.add_scalar("train/single/total_grad_norm", total_norm, iters)

        opt.step()

        mid_str = ''
        if type(model.module) is MultiGPUFastTransformer and args.self_distil > 0.0:
            mid_str += 'distil={:.5f}, '.format(self_distil_loss.cpu().data.numpy()[0])
        if type(model.module) is MultiGPUFastTransformer and "predict" in args.trg_len_option:
            mid_str += 'pred_target_len_loss={:.5f}, '.format(float(pred_target_len_loss.cpu().data.numpy()))
        if type(model.module) is MultiGPUFastTransformer and args.denoising_prob > 0.0:
            mid_str += "/".join(["{:.1f}".format(ff) for ff in denoising_weights[:-1]])+", "

        #mid_str += "data_prep_time={:.5f}, model_time={:.5f}, ".format(t2_data - t1_data, t2_model - t1_model)
        mid_str += ""

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
                if trg_len_dic != None:
                    writer.add_scalar("train/single/average_target_len_correct", getattr(train_metrics_average, "average_target_len_correct"), iters)
                    writer.add_scalar("train/single/average_target_len_approx", getattr(train_metrics_average, "average_target_len_approx"), iters)

            train_metrics.reset()
            if train_metrics_trg is not None:
                train_metrics_trg.reset()
            if train_metrics_average is not None:
                train_metrics_average.reset()

    #torch.save(targetlength.lengths, str(args.data_prefix / "trg_len_dic" / args.dataset[-4:]))
