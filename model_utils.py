import sys
import torch
from torch.autograd import Variable

def linear_attention(source_masks, decoder_masks, decoder_input_how):
    if decoder_input_how == "copy":
        max_src_len = source_masks.size(1)
        max_trg_len = decoder_masks.size(1)

        src_lens = source_masks.sum(-1).float()-1  # batch_size
        trg_lens = decoder_masks.sum(-1).float()-1  # batch_size
        steps = src_lens / trg_lens          # batch_size

        index_s = torch.arange(max_trg_len)  # max_trg_len
        if decoder_masks.is_cuda:
            index_s = index_s.cuda(decoder_masks.get_device())

        index_s = steps.data[:,None] * index_s[None,:] # batch_size X max_trg_len
        index_s = Variable(torch.round(index_s), requires_grad=False).long()
        return index_s
    else:
        sys.exit("not implemented")

def apply_mask(inputs, mask, p=1):
    _mask = mask.long()
    return inputs * _mask + (torch.mul(_mask, -1) + 1 ) * p

def prepare_sources(batch, field, masks=None):
    masks = prepare_masks(batch.src, field) if masks is None else masks
    if type(masks) is not Variable:
        masks = Variable(masks)
    return batch.src, masks

def prepare_targets(batch, field, targets=None, masks=None):
    if targets is None:
        targets = batch.trg[:, 1:].contiguous()
    masks = prepare_masks(targets, field) if masks is None else masks
    if type(masks) is not Variable:
        masks = Variable(masks)
    return targets, masks

def prepare_decoder_inputs(trg_inputs, field, inputs=None, masks=None):
    decoder_inputs = trg_inputs[:, :-1].contiguous()
    decoder_masks = prepare_masks(trg_inputs[:, 1:], field) if masks is None else masks
    # NOTE why [1:], not [:-1]?
    if type(decoder_masks) is not Variable:
        decoder_masks = Variable(decoder_masks)
    return decoder_inputs, decoder_masks

def prepare_masks(inputs, field):
    if inputs.ndimension() == 2:
        masks = (inputs.data != field.vocab.stoi['<pad>']).float()
    else: # NOTE FALSE
        masks = (inputs.data[:, :, field.vocab.stoi['<pad>']] != 1).float()

    return masks
