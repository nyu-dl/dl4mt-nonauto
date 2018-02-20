import os
os.environ['QT_QPA_PLATFORM']='offscreen' # weird can't ipdb with mscoco without this flag
import torch
import numpy as np
from torchtext import data
from torchtext import datasets
from torch.nn import functional as F
from torch.autograd import Variable

import revtok
import logging
import random
import ipdb
import string
import traceback
import math
import uuid
import argparse
import copy
import time
import pickle

from train import train_model
from distill import distill_model
from model import FastTransformer, Transformer, INF, TINY, HighwayBlock, ResidualBlock, NonresidualBlock
from utils import mkdir, organise_trg_len_dic, init_encoder
from data import NormalField, NormalTranslationDataset, MSCOCODataset, data_path
from time import gmtime, strftime
from decode import decode_model

import sys
import itertools
from traceback import extract_tb
from code import interact
from pathlib import Path

parser = argparse.ArgumentParser(description='Train a Transformer / FastTransformer.')

# dataset settings
parser.add_argument('--dataset',     type=str, default='iwslt-ende', choices=['iwslt-ende', 'iwslt-deen', \
                                                                              'wmt15-ende', 'wmt15-deen', \
                                                                              'wmt16-enro', 'wmt16-roen', \
                                                                              'wmt17-enlv', 'wmt17-lven', \
                                                                              'mscoco'])
parser.add_argument('--vocab_size',      type=int, default=40000,  help='limit the train set sentences to this many tokens')

parser.add_argument('--valid_size',   type=int, default=None, help='size of valid dataset (tested on coco only)')
parser.add_argument('--load_vocab',   action='store_true', help='load a pre-computed vocabulary')
parser.add_argument('--load_dataset', action='store_true', default=False, help='load a pre-processed dataset')
parser.add_argument('--save_dataset', action='store_true', default=False, help='save a pre-processed dataset')
parser.add_argument('--max_len',      type=int, default=None,  help='limit the train set sentences to this many tokens')
parser.add_argument('--max_train_data',      type=int, default=None,  help='limit the train set sentences to this many sentences')

# model basic settings
parser.add_argument('--prefix', type=str, default='[time]',      help='prefix to denote the model, nothing or [time]')
parser.add_argument('--fast',   dest='model', action='store_const', const=FastTransformer, default=Transformer)

# model ablation settings
parser.add_argument('--ffw_block',     type=str, default="residual", choices=['residual', 'highway', 'nonresidual'])
parser.add_argument('--diag',   action='store_true', default=False, help='ignore diagonal attention when doing self-attention.')
parser.add_argument('--use_wo',   action='store_true', default=True, help='use output weight matrix in multihead attention')
parser.add_argument('--inputs_dec', type=str, default='pool', choices=['zeros', 'pool'], help='inputs to first decoder')
parser.add_argument('--out_norm', action='store_true', default=False, help='normalize last softmax layer')
parser.add_argument('--share_embed',  action='store_true', default=True, help='share embeddings and linear out weight')
parser.add_argument('--share_vocab',  action='store_true', default=True, help='share vocabulary between src and target')
parser.add_argument('--share_embed_enc_dec1',  action='store_true', default=False, help='share embedding weigth between encoder and first decoder')
parser.add_argument('--positional', action='store_true', default=True, help='incorporate positional information in key/value')
parser.add_argument('--enc_last', action='store_true', default=False, help='attend only to last encoder hidden states')

parser.add_argument('--params', type=str, default='user', choices=['user', 'small', 'big'])
parser.add_argument('--n_layers',  type=int, default=5,    help='number of layers')
parser.add_argument('--n_heads',   type=int,   default=2, help='number of heads')
parser.add_argument('--d_model',   type=int,   default=278, help='number of heads')
parser.add_argument('--d_hidden',  type=int,   default=507, help='number of heads')

parser.add_argument('--num_decs', type=int, default=2, help='1 (one shared decoder) \
                                                                   2 (2nd decoder and above is shared) \
                                                                  -1 (no decoder is shared)')
parser.add_argument('--train_repeat_dec', type=int, default=4, help='number of times to repeat generation')
parser.add_argument('--valid_repeat_dec', type=int, default=4, help='number of times to repeat generation')
parser.add_argument('--use_argmax', action='store_true', default=False)
parser.add_argument('--next_dec_input', type=str, default='both', choices=['emb', 'out', 'both'])

parser.add_argument('--bp',  type=float,   default=1.0, help='number of heads')

# running setting
parser.add_argument('--mode',    type=str, default='train',  choices=['train', 'test', 'distill']) # distill : take a trained AR model and decode a training set
parser.add_argument('--gpu',     type=int, default=0,        help='GPU to use or -1 for CPU')
parser.add_argument('--seed',    type=int, default=19920206, help='seed for randomness')
parser.add_argument('--distill_which',    type=int, default=0 )
parser.add_argument('--decode_which',    type=int, default=0 )
parser.add_argument('--test_which',    type=str, default='test',  choices=['valid', 'test']) # distill : take a trained AR model and decode a training set

# training
parser.add_argument('--no_tqdm',       action="store_true", default=False)
parser.add_argument('--eval_every',    type=int, default=100,    help='run dev every')
parser.add_argument('--save_every',    type=int, default=-1,   help='5000')
parser.add_argument('--batch_size',    type=int, default=2048,    help='# of tokens processed per batch')
parser.add_argument('--optimizer',     type=str, default='Adam')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--lr_schedule', type=str, default='anneal', choices=['transformer', 'anneal', 'fixed'])
parser.add_argument('--warmup', type=int, default=16000, help='maximum steps to linearly anneal the learning rate')
parser.add_argument('--anneal_steps', type=int, default=250000, help='maximum steps to linearly anneal the learning rate')
parser.add_argument('--maximum_steps', type=int, default=5000000, help='maximum steps you take to train a model')
parser.add_argument('--drop_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--drop_len_pred', type=float, default=0.3, help='dropout ratio for length prediction module')
parser.add_argument('--input_drop_ratio', type=float, default=0.1, help='dropout ratio only for inputs')
parser.add_argument('--grad_clip', type=float, default=-1.0, help='gradient clipping')

# target length
parser.add_argument('--trg_len_option',     type=str, default="reference", choices=['reference', "noisy_ref", 'average', 'fixed', 'predict'])
#parser.add_argument('--trg_len_option_valid',     type=str, default="average", choices=['reference', "noisy_ref", 'average', 'fixed', 'predict'])
parser.add_argument('--trg_len_ratio',  type=float,   default=2.0)
parser.add_argument('--decoder_input_how', type=str, default='copy', choices=['copy', 'interpolate', 'pad', 'wrap'])
parser.add_argument('--finetune_trg_len',  action='store_true', default=False, help="finetune one layer that predicts target len offset")
parser.add_argument('--use_predicted_trg_len',  action='store_true', default=False, help="use predicted target len masks")
parser.add_argument('--max_offset', type=int, default=20, help='max target len offset of the whole dataset')

# denoising
parser.add_argument('--denoising_prob',   type=float, default=0.0, help="use denoising with this probability")
parser.add_argument('--denoising_weight',   type=float, default=0.1, help="use denoising with this weight.")
parser.add_argument('--corruption_probs',   type=str, default="0-0-0-1-1-1-0", help="probs for \
                    repeat\
                    add random word\
                    repeat and drop next\
                    replace with random word\
                    swap\
                    global swap")
parser.add_argument('--denoising_out_weight',   type=float, default=0.0, help="use denoising for decoder output with this weight.")
parser.add_argument('--anneal_denoising_weight',       action='store_true', default=False, help="anneal denoising weight over time")
parser.add_argument('--layerwise_denoising_weight',       action='store_true', default=False, help="use different denoising weight per iteration")

# self-distillation
parser.add_argument('--self_distil',   type=float, default=0.0)

# decoding
parser.add_argument('--length_ratio',  type=int,   default=2, help='maximum lengths of decoding')
parser.add_argument('--length_dec',  type=int,   default=20, help='maximum length of decoding for MSCOCO dataset')
parser.add_argument('--beam_size',     type=int,   default=1, help='beam-size used in Beamsearch, default using greedy decoding')
parser.add_argument('--f_size',        type=int,   default=1, help='heap size for sampling/searching in the fertility space')
parser.add_argument('--alpha',         type=float, default=1, help='length normalization weights')
parser.add_argument('--temperature',   type=float, default=1, help='smoothing temperature for noisy decodig')
parser.add_argument('--remove_repeats',       action='store_true', default=False, help='debug mode: no saving or tensorboard')
parser.add_argument('--num_samples', type=int, default=2, help='number of samples to use when using non-argmax decoding')
parser.add_argument('--T', type=float, default=1, help='softmax temperature when decoding')

#parser.add_argument('--jaccard_stop', action='store_true', default=False, help='use jaccard index to stop decoding')
parser.add_argument('--adaptive_decoding', type=str, default=None, choices=["oracle", "jaccard", "equality"])
parser.add_argument('--adaptive_window',     type=int,   default=5, help='window size for adaptive decoding')
parser.add_argument('--len_stop', action='store_true', default=False, help='use length of sentence to stop decoding')
parser.add_argument('--jaccard_thresh', type=float, default=1.0)

# model saving/reloading, output translations
parser.add_argument('--load_from',     type=str, default=None, help='load from checkpoint')
parser.add_argument('--load_encoder_from',     type=str, default=None, help='load from checkpoint')
parser.add_argument('--resume',        action='store_true', help='when loading from the saved model, it resumes from that.')
parser.add_argument('--use_distillation', action='store_true', default=False,     help='train a NAR model from output of an AR model')

# debugging
parser.add_argument('--debug',       action='store_true', help='debug mode: no saving or tensorboard')
parser.add_argument('--tensorboard', action='store_true', default=True, help='use TensorBoard')

# save path
parser.add_argument('--main_path', type=str, default="./") # /misc/vlgscratch2/ChoGroup/mansimov/
parser.add_argument('--model_path', type=str, default="models") # /misc/vlgscratch2/ChoGroup/mansimov/
parser.add_argument('--log_path', type=str, default="logs") # /misc/vlgscratch2/ChoGroup/mansimov/
parser.add_argument('--event_path', type=str, default="events") # /misc/vlgscratch2/ChoGroup/mansimov/
parser.add_argument('--decoding_path', type=str, default="decoding") # /misc/vlgscratch2/ChoGroup/mansimov/
parser.add_argument('--distill_path', type=str, default="distill") # /misc/vlgscratch2/ChoGroup/mansimov/

parser.add_argument('--model_str', type=str, default="") # /misc/vlgscratch2/ChoGroup/mansimov/

# ----------------------------------------------------------------------------------------------------------------- #

args = parser.parse_args()

if args.model is Transformer:
    args.num_decs = 1
    args.train_repeat_dec = 1
    args.valid_repeat_dec = 1

args.main_path = Path(args.main_path)

args.model_path = args.main_path / args.model_path / args.dataset
args.log_path = args.main_path / args.log_path / args.dataset
args.event_path = args.main_path / args.event_path / args.dataset
args.decoding_path = args.main_path / args.decoding_path / args.dataset
args.distill_path = args.main_path / args.distill_path / args.dataset

if not args.debug:
    for path in [args.model_path, args.log_path, args.event_path, args.decoding_path, args.distill_path]:
        path.mkdir(parents=True, exist_ok=True)

if args.prefix == '[time]':
    args.prefix = strftime("%m.%d_%H.%M.", gmtime())

if args.train_repeat_dec == 1:
    args.num_decs = 1

# get the langauage pairs:
if args.dataset != "mscoco":
    args.src = args.dataset[-4:][:2]  # source language
    args.trg = args.dataset[-4:][2:]  # target language
else:
    args.src = ""
    args.trg = ""

if args.params == 'small':
    hparams = {'d_model': 278, 'd_hidden': 507, 'n_layers': 5, 'n_heads': 2, 'warmup': 746}
    args.__dict__.update(hparams)
elif args.params == 'big':
    if args.dataset != "mscoco":
        hparams = {'d_model': 512, 'd_hidden': 512, 'n_layers': 6, 'n_heads': 8, 'warmup': 16000}
    else:
        hparams = {'d_model': 512, 'd_hidden': 512, 'n_heads': 8, 'warmup': 16000}
    args.__dict__.update(hparams)

hp_str = "{}".format('' if args.model is FastTransformer else 'ar_') + \
         "{}".format(args.model_str+"_" if args.model_str != "" else "") + \
         "{}".format("ar_distil_" if args.use_distillation else "") + \
         "{}".format("ptrn_enc_" if not args.load_encoder_from is None else "") + \
         "{}".format("ptrn_model_" if not args.load_from is None else "") + \
         "voc{}k_".format(args.vocab_size//1000) + \
         "{}_".format(args.batch_size) + \
         "{}".format("" if args.share_embed else "no_share_emb_") + \
         "{}".format("" if args.share_vocab else "no_share_voc_") + \
         "{}".format("share_emb_enc_dec1_" if args.share_embed_enc_dec1 else "") + \
         "{}_{}_{}_{}_".format(args.n_layers, args.d_model, args.d_hidden, args.n_heads) + \
         "{}".format("enc_last_" if args.enc_last else "") + \
         "drop_{}_".format(args.drop_ratio) + \
         "{}".format("drop_len_pred_{}_".format(args.drop_len_pred) if args.finetune_trg_len else "") + \
         "{}_".format(args.lr) + \
         "{}_".format("{}".format(args.lr_schedule[:4])) + \
         "{}".format("anneal_steps_{}_".format(args.anneal_steps) if args.lr_schedule == "anneal" else "") + \
         "{}_".format(args.ffw_block[:4]) + \
         "{}".format("clip_{}_".format(args.grad_clip) if args.grad_clip != -1.0 else "") + \
         "{}".format("diag_" if args.diag else "") + \
         ("tr{}_".format(args.train_repeat_dec) + \
         "{}decs_".format(args.num_decs) + \
         "{}_".format(args.bp if args.trg_len_option == "noisy_ref" else "") + \
         "{}_".format(args.trg_len_option[:4]) + \
         "{}_".format(args.next_dec_input) + \
         "{}".format("trg_{}x_".format(args.trg_len_ratio) if "fixed" in args.trg_len_option else "") + \
         "{}_".format(args.decoder_input_how[:4]) + \
         "{}".format("dn_{}_".format(args.denoising_prob) if args.denoising_prob != 0.0 else "") + \
         "{}".format("dn_w{}_".format(args.denoising_weight) if args.denoising_prob != 0.0 and not args.anneal_denoising_weight and not args.layerwise_denoising_weight else "") + \
         "{}".format("dn_anneal_" if args.anneal_denoising_weight else "") + \
         "{}".format("dn_layer_" if args.layerwise_denoising_weight else "") + \
         "{}".format("dn_out_w{}_".format(args.denoising_out_weight) if args.denoising_out_weight != 0.0 else "") + \
         "{}".format("distil{}_".format(args.self_distil) if args.self_distil != 0.0 else "") + \
         "{}".format("argmax_" if args.use_argmax else "sample_") + \
         "{}".format("out_norm_" if args.out_norm else "") + \
         "" if args.model is FastTransformer else "" )

args.id_str = Path(args.prefix + hp_str)

args.corruption_probs = [int(xx) for xx in args.corruption_probs.split("-") ]
c_probs_sum = sum(args.corruption_probs)
args.corruption_probs = [xx/c_probs_sum for xx in args.corruption_probs]

if args.ffw_block == "nonresidual":
    args.block_cls = NonresidualBlock
elif args.ffw_block == "residual":
    args.block_cls = ResidualBlock
elif args.ffw_block == "highway":
    args.block_cls = HighwayBlock
else:
    raise

# setup logger settings
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
if not args.debug:
    fh = logging.FileHandler( str( args.log_path / args.id_str ) + ".txt" )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# setup random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# ----------------------------------------------------------------------------------------------------------------- #
if args.dataset != "mscoco":
    DataField = NormalField
    TRG   = DataField(init_token='<init>', eos_token='<eos>', batch_first=True)
    SRC   = DataField(batch_first=True) if not args.share_vocab else TRG
    # NOTE : UNK, PAD, INIT, EOS

# setup many datasets (need to manaually setup)
data_prefix = Path(data_path(args.dataset))
args.data_prefix = data_prefix
if args.dataset == "mscoco":
    data_prefix = str(data_prefix)

train_dir = "train" if not args.use_distillation else "distill/" + args.dataset[-4:]
if args.dataset == 'iwslt-ende' or args.dataset == 'iwslt-deen':
    if args.resume:
        train_dir += "2"
    logger.info("TRAINING CORPUS : " + str(data_prefix / train_dir / 'train.tags.en-de.bpe'))
    train_data = NormalTranslationDataset(path=str(data_prefix / train_dir / 'train.tags.en-de.bpe'),
    exts=('.{}'.format(args.src), '.{}'.format(args.trg)), fields=(SRC, TRG),
    load_dataset=args.load_dataset, save_dataset=args.save_dataset, prefix='normal') \
        if args.mode in ["train", "distill"] else None

    dev_dir = "dev"
    dev_file = "valid.en-de.bpe"
    if args.mode == "test" and args.decode_which > 0:
        dev_dir = "dev_split"
        dev_file += ".{}".format(args.decode_which)
    dev_data = NormalTranslationDataset(path=str(data_prefix / dev_dir / dev_file),
    exts=('.{}'.format(args.src), '.{}'.format(args.trg)), fields=(SRC, TRG),
    load_dataset=args.load_dataset, save_dataset=args.save_dataset, prefix='normal')

    test_data = None

elif args.dataset == 'wmt15-ende' or args.dataset == 'wmt15-deen':
    train_file = 'all_de-en.bpe'
    if args.mode == "distill" and args.distill_which > 0:
        train_file += ".{}".format(args.distill_which)
    train_data = NormalTranslationDataset(path=str(data_prefix / train_dir / train_file),
    exts=('.{}'.format(args.src), '.{}'.format(args.trg)), fields=(SRC, TRG),
    load_dataset=args.load_dataset, save_dataset=args.save_dataset, prefix='normal') \
        if args.mode in ["train", "distill"] else None

    dev_dir = "dev"
    dev_file = "newstest2013.bpe"
    if args.mode == "test" and args.decode_which > 0:
        dev_dir = "dev_split"
        dev_file += ".{}".format(args.decode_which)
    dev_data = NormalTranslationDataset(path=str(data_prefix / dev_dir / dev_file),
    exts=('.{}'.format(args.src), '.{}'.format(args.trg)), fields=(SRC, TRG),
    load_dataset=args.load_dataset, save_dataset=args.save_dataset, prefix='normal')

    test_dir = "test"
    test_file = "newstest2014-deen.bpe"
    if args.mode == "test" and args.decode_which > 0:
        test_dir = "test_split"
        test_file += ".{}".format(args.decode_which)
    test_data = NormalTranslationDataset(path=str(data_prefix / test_dir / test_file),
    exts=('.{}'.format(args.src), '.{}'.format(args.trg)), fields=(SRC, TRG),
    load_dataset=args.load_dataset, save_dataset=args.save_dataset, prefix='normal')

elif args.dataset == 'wmt16-enro' or args.dataset == 'wmt16-roen':
    train_file = 'corpus.bpe'
    if args.mode == "distill" and args.distill_which > 0:
        train_file += ".{}".format(args.distill_which)
    train_data = NormalTranslationDataset(path=str(data_prefix / train_dir / train_file),
    exts=('.{}'.format(args.src), '.{}'.format(args.trg)), fields=(SRC, TRG),
    load_dataset=args.load_dataset, save_dataset=args.save_dataset, prefix='normal') \
        if args.mode in ["train", "distill"] else None

    dev_dir = "dev"
    dev_file = "dev.bpe"
    if args.mode == "test" and args.decode_which > 0:
        dev_dir = "dev_split"
        dev_file += ".{}".format(args.decode_which)
    dev_data = NormalTranslationDataset(path=str(data_prefix / dev_dir / dev_file),
    exts=('.{}'.format(args.src), '.{}'.format(args.trg)), fields=(SRC, TRG),
    load_dataset=args.load_dataset, save_dataset=args.save_dataset, prefix='normal')

    test_dir = "test"
    test_file = "test.bpe"
    if args.mode == "test" and args.decode_which > 0:
        test_dir = "test_split"
        test_file += ".{}".format(args.decode_which)
    test_data = NormalTranslationDataset(path=str(data_prefix / test_dir / test_file),
    exts=('.{}'.format(args.src), '.{}'.format(args.trg)), fields=(SRC, TRG),
    load_dataset=args.load_dataset, save_dataset=args.save_dataset, prefix='normal')

elif args.dataset == 'wmt17-enlv' or args.dataset == 'wmt17-lven':
    train_data, dev_data, test_data = NormalTranslationDataset.splits(
    path=data_prefix, train='{}/corpus.bpe'.format(train_dir), test='test/newstest2017.bpe',
    validation='dev/newsdev2017.bpe', exts=('.{}'.format(args.src), '.{}'.format(args.trg)),
    fields=(SRC, TRG), load_dataset=args.load_dataset, save_dataset=args.save_dataset, prefix='normal')

elif args.dataset == "mscoco":
    mscoco_dataset = MSCOCODataset(path=data_prefix, batch_size=args.batch_size, \
                        max_len=args.max_len, valid_size=args.valid_size, \
                        distill=(args.mode == "distill"), use_distillation=args.use_distillation)
    train_data, train_sampler = mscoco_dataset.train_data, mscoco_dataset.train_sampler
    dev_data, dev_sampler = mscoco_dataset.valid_data, mscoco_dataset.valid_sampler
    test_data, test_sampler = mscoco_dataset.test_data, mscoco_dataset.test_sampler
    if args.trg_len_option == "predict" and args.max_offset == None:
        args.max_offset = mscoco_dataset.max_dataset_length
else:
    raise NotImplementedError

# build vocabularies for translation dataset
if args.dataset != "mscoco":
    vocab_path = data_prefix / 'vocab' / '{}_{}_{}.pt'.format('{}-{}'.format(args.src, args.trg), args.vocab_size, 'shared' if args.share_vocab else '')
    if args.load_vocab and vocab_path.exists():
        src_vocab, trg_vocab = torch.load(str(vocab_path))
        SRC.vocab = src_vocab
        TRG.vocab = trg_vocab
        logger.info('vocab loaded')
    else:
        assert (not train_data is None)
        if not args.share_vocab:
            SRC.build_vocab(train_data, dev_data, max_size=args.vocab_size)
        TRG.build_vocab(train_data, dev_data, max_size=args.vocab_size)
        if not args.debug and not args.use_distillation:
            logger.info('save the vocabulary')
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save([SRC.vocab, TRG.vocab], str(vocab_path))
    args.__dict__.update({'trg_vocab': len(TRG.vocab), 'src_vocab': len(SRC.vocab)})
# for mscoco
else:
    vocab_path = os.path.join(data_prefix, "vocab.pkl")
    assert (args.load_vocab == True)
    if args.load_vocab and os.path.exists(vocab_path):
        vocab = pickle.load(open(vocab_path, 'rb'))
        mscoco_dataset.vocab = vocab
    else:
        logger.info('save the vocabulary')
        mscoco_dataset.build_vocab()
        pickle.dump(mscoco_dataset.vocab, open(vocab_path, 'wb'))
        print ('vocab building done')
    args.__dict__.update({'vocab': len(mscoco_dataset.vocab)})

def dyn_batch_with_padding(new, i, sofar):
    prev_max_len = sofar / (i - 1) if i > 1 else 0
    return max(len(new.src), len(new.trg),  prev_max_len) * i

def dyn_batch_without_padding(new, i, sofar):
    return sofar + max(len(new.src), len(new.trg))

# not sure if absolutely necessary? seems to mess things up.
if args.dataset != "mscoco" and args.share_vocab:
    SRC = copy.deepcopy(SRC)
    SRC.init_token = None
    SRC.eos_token = None

    for data_ in [train_data, dev_data, test_data]:
        if not data_ is None:
            data_.fields['src'] = SRC

if args.dataset != "mscoco":
    if not train_data is None:
        logger.info("before pruning : {} training examples".format(len(train_data.examples)))
        if args.max_len is not None:
            if args.dataset != "mscoco":
                train_data.examples = [ex for ex in train_data.examples if len(ex.trg) <= args.max_len]
        if args.max_train_data is not None:
            train_data.examples = train_data.examples[:args.max_train_data]
        logger.info("after pruning : {} training examples".format(len(train_data.examples)))

if args.batch_size == 1:  # speed-test: one sentence per batch.
    batch_size_fn = lambda new, count, sofar: count
else:
    batch_size_fn = dyn_batch_without_padding if args.model is Transformer else dyn_batch_with_padding

if args.dataset != "mscoco":
    if args.mode == "train":
        train_flag = True
    elif args.mode == "distill":
        train_flag = False
    else:
        train_flag = False
    train_real = data.BucketIterator(train_data, args.batch_size, device=args.gpu, batch_size_fn=batch_size_fn,
                                    train=train_flag, repeat=train_flag, shuffle=train_flag) if not train_data is None else None
    dev_real = data.BucketIterator(dev_data, args.batch_size, device=args.gpu, batch_size_fn=batch_size_fn,
                                    train=False, repeat=False, shuffle=False) if not dev_data is None else None
    test_real = data.BucketIterator(test_data, args.batch_size, device=args.gpu, batch_size_fn=batch_size_fn,
                                    train=False, repeat=False, shuffle=False) if not test_data is None else None
else:
    train_real = torch.utils.data.DataLoader(
        train_data, batch_sampler=train_sampler, pin_memory=args.gpu>-1, num_workers=8)
    dev_real = torch.utils.data.DataLoader(
        dev_data, batch_sampler=dev_sampler, pin_memory=args.gpu>-1, num_workers=8)
    test_real = torch.utils.data.DataLoader(
        test_data, batch_sampler=test_sampler, pin_memory=args.gpu>-1, num_workers=8)
    def rcycle(iterable):
        saved = []                 # In-memory cache
        for element in iterable:
            yield element
            saved.append(element)
        while saved:
            random.shuffle(saved)  # Shuffle every batch
            for element in saved:
                  yield element
    if args.mode != "distill":
        train_real = rcycle(train_real)

logger.info("build the dataset. done!")
# ----------------------------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------------------------- #
if args.mode == "train":
    logger.info(args)

logger.info('Starting with HPARAMS: {}'.format(hp_str))

# build the model
if args.dataset != "mscoco":
    model = args.model(src=SRC, trg=TRG, args=args)
else:
    model = args.model(src=None, trg=mscoco_dataset, args=args)

if args.mode == "train":
    logger.info(str(model))

if args.load_encoder_from is not None:
    if args.gpu > -1:
        with torch.cuda.device(args.gpu):
            encoder = torch.load(str(args.model_path / args.load_encoder_from) + '.pt',
                                 map_location=lambda storage, loc: storage.cuda())
    else:
        encoder = torch.load(str(args.model_path / args.load_encoder_from) + '.pt',
                             map_location=lambda storage, loc: storage)
    init_encoder(model, encoder)
    logger.info("Pretrained encoder loaded.")

if args.load_from is not None:
    if args.gpu > -1:
        with torch.cuda.device(args.gpu):
            model.load_state_dict(torch.load(str(args.model_path / args.load_from) + '.pt',
            map_location=lambda storage, loc: storage.cuda()), strict=False)  # load the pretrained models.
    else:
        model.load_state_dict(torch.load(str(args.model_path / args.load_from) + '.pt',
        map_location=lambda storage, loc: storage), strict=False)  # load the pretrained models.
    logger.info("Pretrained model loaded.")

params, param_names = [], []
for name, param in model.named_parameters():
    params.append(param)
    param_names.append(name)

if args.mode == "train":
    logger.info(param_names)
    logger.info("Size {}".format( sum( [ np.prod(x.size()) for x in params ] )) )

# use cuda
if args.gpu > -1:
    model.cuda(args.gpu)

# additional information
args.__dict__.update({'hp_str': hp_str,  'logger': logger})

# ----------------------------------------------------------------------------------------------------------------- #

trg_len_dic = None
if args.dataset != "mscoco" and (not "ro" in args.dataset or "predict" in args.trg_len_option or "average" in args.trg_len_option):
#if "predict" in args.trg_len_option or "average" in args.trg_len_option:
    #trg_len_dic = torch.load(os.path.join(data_path(args.dataset), "trg_len"))
    trg_len_dic = torch.load( str(args.data_prefix / "trg_len_dic" / args.dataset[-4:]) )
    trg_len_dic = organise_trg_len_dic(trg_len_dic)

if args.mode == 'train':
    logger.info('starting training')

    if args.dataset != "mscoco":
        train_model(args, model, train_real, dev_real, src=SRC, trg=TRG, trg_len_dic=trg_len_dic)
    else:
        train_model(args, model, train_real, dev_real, src=None, trg=mscoco_dataset, trg_len_dic=trg_len_dic)

elif args.mode == 'test':
    logger.info('starting decoding from the pre-trained model, on the test set...')
    args.decoding_path = args.decoding_path / args.load_from
    name_suffix = 'b={}_{}.txt'.format(args.beam_size, args.load_from)
    names = ['src.{}'.format(name_suffix), 'trg.{}'.format(name_suffix), 'dec.{}'.format(name_suffix)]

    if args.test_which == "test" and (not test_real is None):
        logger.info("---------- Decoding TEST set ----------")
        decode_model(args, model, test_real, evaluate=True, trg_len_dic=trg_len_dic, decoding_path=args.decoding_path, \
                     names=["test."+xx for xx in names], maxsteps=None)
    else:
        logger.info("---------- Decoding VALID set ----------")
        decode_model(args, model, dev_real, evaluate=True, trg_len_dic=trg_len_dic, decoding_path=args.decoding_path, \
                     names=["valid."+xx for xx in names], maxsteps=None)

elif args.mode == 'distill':
    logger.info('starting decoding the training set from an AR model')
    args.distill_path = args.distill_path / args.id_str
    args.distill_path.mkdir(parents=True, exist_ok=True)
    name_suffix = args.distill_which
    names = ['src.{}'.format(name_suffix), 'trg.{}'.format(name_suffix), 'dec.{}'.format(name_suffix)]

    distill_model(args, model, train_real, evaluate=False, distill_path=args.distill_path, \
                  names=["train."+xx for xx in names], maxsteps=None)

logger.info("done.")
