import torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
import _pickle as pickle
import json
import numpy as np
import time
import random
from collections import OrderedDict

def process_json(dataPath, annFile, max_len=None, size=None):
    annPath = os.path.join(dataPath, annFile)

    # load dataset
    annots = json.load(open(annPath, 'r'))
    if size != None:
        annots = annots[:size]

    bpes = []
    features_path = []
    bpe2img = {}
    img2bpes = {}

    bpe_i, feature_i = 0, 0

    for annot in annots:
        bpes_i = []
        for bpe in annot['bpes']:
            len_bpe = len(bpe.split(' '))
            if max_len != None and len_bpe > max_len:
                continue
            bpes.append(bpe)
            bpe2img[bpe_i] = feature_i
            bpes_i.append(bpe_i)
            bpe_i = bpe_i + 1
        img2bpes[feature_i] = bpes_i
        img_name = annot['img_name'] + '.npy'
        if 'train' in img_name:
            load_path = os.path.join(dataPath, 'train2014_features')
        elif 'val' in img_name:
            load_path = os.path.join(dataPath, 'val2014_features')
        else:
            sys.exit()
        features_path.append(os.path.join(load_path, img_name))
        feature_i = feature_i + 1

    return bpes, features_path, bpe2img, img2bpes

def minibatch_same_length(lengths, batch_size):
    # make sure all of them are integers
    all(isinstance(ll, int) for ll in lengths)

    # sort them out
    len_unique = np.unique(lengths)

    # indices of unique lengths
    len_indices = OrderedDict()
    len_counts = OrderedDict()
    for ll in len_unique:
        len_indices[ll] = np.where(lengths == ll)[0]
        len_counts[ll] = len(len_indices[ll])

    # sort indicies into minibatches
    minibatches = []
    len_indices_keys = list(len_indices.keys())
    for k in len_indices_keys:
        avg_samples = max(1, int(batch_size / k))
        for j in range(0, len_counts[k], avg_samples):
            minibatches.append(len_indices[k][j:j+avg_samples])

    return minibatches

class BatchSamplerCaptionsSameLength(object):
    def __init__(self, dataset, batch_size):
        assert (type(dataset) == CocoCaptionsIndexedCaption)
        self.bpes = dataset.bpes
        lengths = []

        for bpe in self.bpes:
            len_bpe = len(bpe.split(' '))
            lengths.append(len_bpe)

        self.minibatches = minibatch_same_length(lengths, batch_size)
        random.shuffle(self.minibatches)

    def __iter__(self):
        # randomly sample minibatch index
        for i in range(len(self.minibatches)):
            minibatch = self.minibatches[i]
            yield minibatch

    def __len__(self):
        return len(self.minibatches)

class BatchSamplerImagesSameLength(object):
    def __init__(self, dataset, batch_size):
        assert (type(dataset) == CocoCaptionsIndexedImage or type(dataset) == CocoCaptionsIndexedImageDistill)
        self.img2bpes = dataset.img2bpes
        self.bpes = dataset.bpes

        # calculate average length of 5 captions for each image
        lengths = []
        img_keys = self.img2bpes.keys()
        for i in img_keys:
            length_i = []
            for bpe_i in self.img2bpes[i]:
                length_i.append(len(self.bpes[bpe_i].split()))
            lengths.append(int(np.mean(np.array(length_i))))

        self.minibatches = minibatch_same_length(lengths, batch_size)
        random.shuffle(self.minibatches)


    def __iter__(self):
        # randomly sample minibatch index
        for i in range(len(self.minibatches)):
            minibatch = self.minibatches[i]
            yield minibatch

    def __len__(self):
        return len(self.minibatches)

# dataset indexed based on images
class CocoCaptionsIndexedImage(data.Dataset):
    def __init__(self, bpes, features_path, bpe2img, img2bpes):
        self.bpes = bpes
        self.features_path = features_path
        self.bpe2img = bpe2img
        self.img2bpes = img2bpes

    def __getitem__(self, index):
        feature = np.float32(np.load(self.features_path[index]))
        bpes = []
        for i in self.img2bpes[index]:
            bpes.append(self.bpes[i])
        return torch.from_numpy(feature), bpes

    def __len__(self):
        return len(self.img2bpes.keys())

class CocoCaptionsIndexedImageDistill(data.Dataset):
    def __init__(self, bpes, features_path, bpe2img, img2bpes):
        self.bpes = bpes
        self.features_path = features_path
        self.bpe2img = bpe2img
        self.img2bpes = img2bpes

    def __getitem__(self, index):
        feature = np.float32(np.load(self.features_path[index]))
        img_name = self.features_path[index].split('/')[-1].split('.')[0]
        bpes = []
        for i in self.img2bpes[index]:
            bpes.append(self.bpes[i])
        return torch.from_numpy(feature), bpes, img_name

    def __len__(self):
        return len(self.img2bpes.keys())

# dataset indexed based on captions
class CocoCaptionsIndexedCaption(data.Dataset):
    def __init__(self, bpes, features_path, bpe2img, img2bpes):
        self.bpes = bpes
        self.features_path = features_path
        self.bpe2img = bpe2img
        self.img2bpes = img2bpes

    def __getitem__(self, index):
        bpe = self.bpes[index]
        feature = np.float32(np.load(self.features_path[self.bpe2img[index]]))
        return torch.from_numpy(feature), bpe

    def __len__(self):
        return len(self.bpe2img.keys())
