import math
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
from mscoco import CocoCaptionsIndexedImage, CocoCaptionsIndexedCaption, CocoCaptionsIndexedImageDistill, \
                    BatchSamplerImagesSameLength, BatchSamplerCaptionsSameLength
from mscoco import process_json

try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction

def data_path(dataset):
    if dataset == "iwslt-ende" or dataset == "iwslt-deen":
        path="IWSLT/en-de/"
    elif dataset == "wmt15-ende" or dataset == "wmt15-deen":
        path="wmt15/deen_new/"
    elif dataset == "wmt14-ende" or dataset == "wmt14-deen":
        path="wmt14/en-de/"
    elif dataset == "wmt16-enro" or dataset == "wmt16-roen":
        path="wmt16/en-ro/"
    elif dataset == "wmt17-enlv" or dataset == "wmt17-lven":
        path="wmt17/en-lv/"
    elif dataset == "mscoco":
        path="mscoco"

    if "vine" in os.uname()[1] \
       or "weaver" in os.uname()[1] \
       or "dgx" in os.uname()[1] \
       or "lion" in os.uname()[1]:
        if dataset != "mscoco":
            if dataset == "iwslt-ende" or dataset == "iwslt-deen":
                return "/misc/kcgscratch1/ChoGroup/jason/{}".format(path)
            else:
                return "/misc/kcgscratch1/ChoGroup/jason/corpora/{}".format(path)
        else:
            return "/misc/kcgscratch1/ChoGroup/mansimov/{}".format(path)
    else:
        if dataset != "mscoco":
            return "/scratch/yl1363/{}".format(path)
        else:
            return "/scratch/em3382/{}".format(path)


# load the dataset + reversible tokenization
class NormalField(data.Field):

    def reverse(self, batch, unbpe=True):
        if not self.batch_first:
            batch.t_()

        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch] # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch] # trim past frst eos
        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        if unbpe:
            batch = [" ".join(filter(filter_special, ex)).replace("@@ ","") for ex in batch]
        else:
            batch = [" ".join(filter(filter_special, ex)) for ex in batch]
        return batch

class MSCOCOVocab(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.stoi:
            self.stoi[word] = self.idx
            self.itos[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.stoi:
            return self.stoi['<unk>']
        return self.stoi[word]

    def __len__(self):
        return len(self.stoi)

class MSCOCODataset(object):
    def __init__(self, path, batch_size, max_len=None, valid_size=None, distill=False, use_distillation=False):
        self.path = path

        if distill:
            self.train_data, self.train_sampler = self.prepare_distill_data(path, 'karpathy_split/train.json.bpe.fixed', batch_size, max_len=max_len, size=None)
        else:
            train_f = 'karpathy_split/train.json.bpe.fixed'
            if use_distillation:
                train_f = 'karpathy_split/train.json.bpe.fixed.high.distill'
            self.train_data, self.train_sampler = self.prepare_train_data(path, train_f, batch_size, max_len=max_len, size=None)

        self.valid_data, self.valid_sampler = self.prepare_test_data(path, 'karpathy_split/valid.json.bpe.fixed', batch_size, max_len=None, size=valid_size)
        self.test_data, self.test_sampler = self.prepare_test_data(path, 'karpathy_split/test.json.bpe.fixed', batch_size, max_len=None, size=valid_size)

        self.unk_token = 0
        self.pad_token = 1
        self.init_token = 2
        self.eos_token = 3

    def prepare_train_data(self, dataPath, annFile, batch_size, max_len=None, size=None):
        bpes, features_path, bpe2img, img2bpes = process_json(dataPath, annFile, max_len=max_len, size=size)

        # get max len of dataset
        self.max_dataset_length = 0
        for bpe in bpes:
            len_bpe = len(bpe.split(' '))
            if len_bpe > self.max_dataset_length:
                self.max_dataset_length = len_bpe

        dataset_captions = CocoCaptionsIndexedCaption(bpes, features_path, bpe2img, img2bpes)
        sampler_captions = BatchSamplerCaptionsSameLength(dataset_captions, batch_size=batch_size)
        return dataset_captions, sampler_captions

    def prepare_test_data(self, dataPath, annFile, batch_size, max_len=None, size=None):
        bpes, features_path, bpe2img, img2bpes = process_json(dataPath, annFile, max_len=max_len, size=size)

        dataset_images = CocoCaptionsIndexedImage(bpes, features_path, bpe2img, img2bpes)
        sampler_images = BatchSamplerImagesSameLength(dataset_images, batch_size=batch_size)
        return dataset_images, sampler_images

    def prepare_distill_data(self, dataPath, annFile, batch_size, max_len=None, size=None):
        bpes, features_path, bpe2img, img2bpes = process_json(dataPath, annFile, max_len=max_len, size=size)

        dataset_images = CocoCaptionsIndexedImageDistill(bpes, features_path, bpe2img, img2bpes)
        sampler_images = BatchSamplerImagesSameLength(dataset_images, batch_size=batch_size)
        return dataset_images, sampler_images


    def build_vocab(self):
        """Build a simple vocabulary wrapper."""
        from collections import Counter

        bpes = self.train_data.bpes

        counter = Counter()
        for bpe in bpes:
            counter.update(bpe.split())

        words = [word for word, cnt in counter.items()]

        # Creates a vocab wrapper and add some special tokens.
        # MAKE SURE CONSTANTS ARE CONSISTENT WITH TRANSLATION DATASETS !!!
        self.vocab = MSCOCOVocab()
        self.vocab.add_word('<unk>')
        self.vocab.add_word('<pad>')
        self.vocab.add_word('<init>')
        self.vocab.add_word('<eos>')

        # Adds the words to the vocabulary.
        for i, word in enumerate(words):
            self.vocab.add_word(word)

    def reverse(self, batch, unbpe=True):
        #batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, '<eos>') for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in ('<init>', '<pad>')

        #batch = [filter(filter_special, ex) for ex in batch]
        if unbpe:
            batch = [" ".join(filter(filter_special, ex)).replace("@@ ","") for ex in batch]
        else:
            batch = [" ".join(filter(filter_special, ex)) for ex in batch]
        return batch

class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, **kwargs):
        """Create a TranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path, exts, fields, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.
        Arguments:
            root: Root dataset storage directory. Default is '.data'.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        #path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class NormalTranslationDataset(TranslationDataset):
    """Defines a dataset for machine translation."""

    def __init__(self, path, exts, fields, load_dataset=False, save_dataset=False, prefix='', **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
        if load_dataset and (os.path.exists(path + '.processed.{}.pt'.format(prefix))):
            examples = pickle.load(open(path + '.processed.{}.pt'.format(prefix), "rb"))
            print ("Loaded TorchText dataset")
        else:
            examples = []
            with open(src_path) as src_file, open(trg_path) as trg_file:
                for src_line, trg_line in zip(src_file, trg_file):
                    src_line, trg_line = src_line.strip(), trg_line.strip()
                    if src_line != '' and trg_line != '':
                        examples.append(data.Example.fromlist(
                            [src_line, trg_line], fields))
            if save_dataset:
                pickle.dump(examples, open(path + '.processed.{}.pt'.format(prefix), "wb"))
                print ("Saved TorchText dataset")

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

class TripleTranslationDataset(datasets.TranslationDataset):
    """Define a triple-translation dataset: src, trg, dec(output of a pre-trained teacher)"""

    def __init__(self, path, exts, fields, load_dataset=False, prefix='', **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]), ('dec', fields[2])]

        src_path, trg_path, dec_path = tuple(os.path.expanduser(path + x) for x in exts)
        if load_dataset and (os.path.exists(path + '.processed.{}.pt'.format(prefix))):
            examples = torch.load(path + '.processed.{}.pt'.format(prefix))
        else:
            examples = []
            with open(src_path) as src_file, open(trg_path) as trg_file, open(dec_path) as dec_file:
                for src_line, trg_line, dec_line in zip(src_file, trg_file, dec_file):
                    src_line, trg_line, dec_line = src_line.strip(), trg_line.strip(), dec_line.strip()
                    if src_line != '' and trg_line != '' and dec_line != '':
                        examples.append(data.Example.fromlist(
                            [src_line, trg_line, dec_line], fields))
            if load_dataset:
                torch.save(examples, path + '.processed.{}.pt'.format(prefix))

        super(datasets.TranslationDataset, self).__init__(examples, fields, **kwargs)

class ParallelDataset(datasets.TranslationDataset):
    """ Define a N-parallel dataset: supports abitriry numbers of input streams"""

    def __init__(self, path=None, exts=None, fields=None,
                load_dataset=False, prefix='', examples=None, **kwargs):

        if examples is None:
            assert len(exts) == len(fields), 'N parallel dataset must match'
            self.N = len(fields)

            paths = tuple(os.path.expanduser(path + x) for x in exts)
            if load_dataset and (os.path.exists(path + '.processed.{}.pt'.format(prefix))):
                examples = torch.load(path + '.processed.{}.pt'.format(prefix))
            else:
                examples = []
                with ExitStack() as stack:
                    files = [stack.enter_context(open(fname)) for fname in paths]
                    for lines in zip(*files):
                        lines = [line.strip() for line in lines]
                        if not any(line == '' for line in lines):
                            examples.append(data.Example.fromlist(lines, fields))
                if load_dataset:
                    torch.save(examples, path + '.processed.{}.pt'.format(prefix))

        super(datasets.TranslationDataset, self).__init__(examples, fields, **kwargs)
