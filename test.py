import sys
import ipdb
import torch
import numpy as np
from torch.autograd import Variable
from utils import corrupt_target

def convert(lst):
    vocab = "what I 've come to realize about Afghanistan , and this is something that is often dismissed in the West".split()
    dd = {idx+4 : word for idx, word in enumerate(vocab)}
    dd[0] = "UNK"
    dd[1] = "PAD"
    dd[2] = "BOS"
    dd[3] = "EOS"
    return " ".join( dd[xx] for xx in lst )

trg             = [ [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 3, 1, 1] ]
decoder_masks   = [ [1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1, 1, 0, 0] ]
weight = float(sys.argv[1])

cor_p = sys.argv[2] # repeat / drop / repeat and drop next / swap / add random word
cor_p = [int(xx) for xx in cor_p.split("-")]
cor_p = [xx/sum(cor_p) for xx in cor_p]

trg = Variable( torch.from_numpy( np.array( trg ) ) )
decoder_masks = torch.from_numpy( np.array( decoder_masks ) )

print ( convert( trg.data.numpy().tolist()[0] ) )
print ( convert( corrupt_target( trg, decoder_masks, 15, weight, cor_p ).data.numpy().tolist()[0] ) )
