import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.init as init
from torch.nn import functional as F
from torch.autograd import Variable, Function
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import math
import random

from utils import computeGLEU, masked_sort, unsorted, make_decoder_masks, query_trg_len_dic

INF = 1e10
TINY = 1e-9

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

def positional_encodings_like(x, t=None):   # hope to be differentiable
    if t is None:
        positions = torch.arange(0, x.size(-2)) # .expand(*x.size()[:2])
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
        positions = Variable(positions.float())
    else:
        positions = t

    # channels
    channels = torch.arange(0, x.size(-1), 2) / x.size(-1) # 0 2 4 6 ... (256)
    if x.is_cuda:
        channels = channels.cuda(x.get_device())
    channels = 1 / (10000 ** Variable(channels))

    # get the positional encoding: batch x target_len
    encodings = positions.unsqueeze(-1) @ channels.unsqueeze(0)  # batch x target_len x 256
    encodings = torch.cat([torch.sin(encodings).unsqueeze(-1), torch.cos(encodings).unsqueeze(-1)], -1)
    encodings = encodings.contiguous().view(*encodings.size()[:-2], -1)  # batch x target_len x 512

    if encodings.ndimension() == 2:
        encodings = encodings.unsqueeze(0).expand_as(x)

    return encodings

class Linear(nn.Linear):
    def __init__(self, d_in, d_out, bias=True, out_norm=False):
        super().__init__(d_in, d_out, bias)
        self.out_norm = out_norm
        stdv = 1. / math.sqrt(self.weight.size(1))
        init.uniform(self.weight, -stdv, stdv)
        if bias:
            self.bias.data.zero_()

    def forward(self, x):
        size = x.size()
        if self.out_norm:
            weight = self.weight / (1e-6 + torch.sqrt((self.weight ** 2).sum(0, keepdim=True)))
            x_ = x / (1e-6 + torch.sqrt((x ** 2).sum(-1, keepdim=True)))
            logit_ = torch.mm(x_.contiguous().view(-1, size[-1]), weight.t()).view(*size[:-1], -1)
            if self.bias:
                logit_ = logit_ + self.bias
            return logit_
        return super().forward(
            x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)

def demask(inputs, the_mask):
    # inputs: 1-D sequences
    # the_mask: batch x max-len
    outputs = Variable((the_mask == 0).long().view(-1))  # 1-D
    indices = torch.arange(0, outputs.size(0))
    if inputs.is_cuda:
        indices = indices.cuda(inputs.get_device())
    indices = indices.view(*the_mask.size()).long()
    indices = indices[the_mask]
    outputs[indices] = inputs
    return outputs.view(*the_mask.size())

# F.softmax has strange default behavior, normalizing over dim 0 for 3D inputs
def softmax(x, T=1):
    return F.softmax(x/T, dim=-1)
    """
    if x.dim() == 3:
        return F.softmax(x.transpose(0, 2)).transpose(0, 2)
    return F.softmax(x)
    """

def log_softmax(x):
    if x.dim() == 3:
        return F.log_softmax(x.transpose(0, 2)).transpose(0, 2)
    return F.log_softmax(x)

def logsumexp(x, dim=-1):
    x_max = x.max(dim, keepdim=True)[0]
    return torch.log(torch.exp(x - x_max.expand_as(x)).sum(dim, keepdim=True) + TINY) + x_max

def gumbel_softmax(input, beta=0.5, tau=1.0):
    noise = input.data.new(*input.size()).uniform_()
    noise.add_(TINY).log_().neg_().add_(TINY).log_().neg_()
    return softmax((input + beta * Variable(noise)) / tau)

# (4, 3, 2) @ (4, 2) -> (4, 3)
# (4, 3) @ (4, 3, 2) -> (4, 3)
# (4, 3, 2) @ (4, 2, 4) -> (4, 3, 4)
def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-1)).squeeze(-1)

def pad_to_match(x, y):
    x_len, y_len = x.size(1), y.size(1)
    if x_len == y_len:
        return x, y
    add_to = x if x_len < y_len else y
    fill = 1 if add_to.dim() == 2 else 0
    extra = add_to.data.new(
        x.size(0), abs(y_len - x_len), *add_to.size()[2:]).fill_(fill)
    if x_len < y_len:
        return torch.cat((x, extra), 1), y
    return x, torch.cat((y, extra), 1)

# --- Top K search with PQ
def topK_search(logits, mask_src, N=100):
    # prepare data
    nlogP = -log_softmax(logits).data
    maxL = nlogP.size(-1)
    overmask = torch.cat([mask_src[:, :, None],
                        (1 - mask_src[:, :, None]).expand(*mask_src.size(), maxL-1) * INF
                        + mask_src[:, :, None]], 2)
    nlogP = nlogP * overmask

    batch_size, src_len, L = logits.size()
    _, R = nlogP.sort(-1)

    def get_score(data, index):
        # avoid all zero
        # zero_mask = (index.sum(-2) == 0).float() * INF
        return data.gather(-1, index).sum(-2)

    heap_scores = torch.ones(batch_size, N) * INF
    heap_inx = torch.zeros(batch_size, src_len, N).long()
    heap_scores[:, :1] = get_score(nlogP, R[:, :, :1])
    if nlogP.is_cuda:
        heap_scores = heap_scores.cuda(nlogP.get_device())
        heap_inx = heap_inx.cuda(nlogP.get_device())

    def span(ins):
        inds = torch.eye(ins.size(1)).long()
        if ins.is_cuda:
            inds = inds.cuda(ins.get_device())
        return ins[:, :, None].expand(ins.size(0), ins.size(1), ins.size(1)) + inds[None, :, :]

    # iteration starts
    for k in range(1, N):
        cur_inx = heap_inx[:, :, k-1]
        I_t = span(cur_inx).clamp(0, L-1)  # B x N x N
        S_t = get_score(nlogP, R.gather(-1, I_t))
        S_t, _inx = torch.cat([heap_scores[:, k:], S_t], 1).sort(1)
        S_t[:, 1:] += ((S_t[:, 1:] - S_t[:, :-1]) == 0).float() * INF  # remove duplicates
        S_t, _inx2 = S_t.sort(1)
        I_t = torch.cat([heap_inx[:, :, k:], I_t], 2).gather(
                        2, _inx.gather(1, _inx2)[:, None, :].expand(batch_size, src_len, _inx.size(-1)))
        heap_scores[:, k:] = S_t[:, :N-k]
        heap_inx[:, :, k:] = I_t[:, :, :N-k]

    # get the searched
    output = R.gather(-1, heap_inx)
    output = output.transpose(2, 1).contiguous().view(batch_size * N, src_len)  # (B x N) x Ts
    output = Variable(output)
    mask_src = mask_src[:, None, :].expand(batch_size, N, src_len).contiguous().view(batch_size * N, src_len)

    return output, mask_src

class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal, diag=False):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal
        self.diag = diag

    def forward(self, query, key, value=None, mask=None,
                feedback=None, beta=0, tau=1, weights=None):
        dot_products = matmul(query, key.transpose(1, 2))   # batch x trg_len x trg_len

        if weights is not None:
            dot_products = dot_products + weights   # additive bias

        if query.dim() == 3 and self.causal and (query.size(1) == key.size(1)):
            tri = key.data.new(key.size(1), key.size(1)).fill_(1).triu(1) * INF
            dot_products.data.sub_(tri.unsqueeze(0))

        if self.diag:
            inds = torch.arange(0, key.size(1)).long().view(1, 1, -1)
            if key.is_cuda:
                inds = inds.cuda(key.get_device())
            dot_products.data.scatter_(1, inds.expand(dot_products.size(0), 1, inds.size(-1)), -INF)
            # eye = key.data.new(key.size(1), key.size(1)).fill_(1).eye() * INF
            # dot_products.data.sub_(eye.unsqueeze(0))

        if mask is not None:
            if type(mask) is Variable:
                mask = mask.data

            if dot_products.dim() == 2:
                dot_products.data -= ((1 - mask) * INF)
            else:
                dot_products.data -= ((1 - mask[:, None, :]) * INF)

        if value is None:
            return dot_products

        logits = dot_products / self.scale
        probs = softmax(logits)

        if feedback is not None:
            feedback.append(probs.contiguous())

        return matmul(self.dropout(probs), value)

class MultiHead2(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio,
                causal=False, diag=False, use_wo=True):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal, diag=diag)
        self.wq = Linear(d_key, d_key, bias=use_wo)
        self.wk = Linear(d_key, d_key, bias=use_wo)
        self.wv = Linear(d_value, d_value, bias=use_wo)
        if use_wo:
            self.wo = Linear(d_value, d_key, bias=use_wo)
        self.use_wo = use_wo
        self.n_heads = n_heads

    def forward(self, query, key, value, mask=None, feedback=None, weights=None, beta=0, tau=1):
        # query : B x T1 x D
        # key : B x T2 x D
        # value : B x T2 x D
        query, key, value = self.wq(query), self.wk(key), self.wv(value)   # B x T x D
        B, Tq, D = query.size()
        _, Tk, _ = key.size()
        N = self.n_heads
        probs = []

        query, key, value = (x.contiguous().view(B, -1, N, D//N).transpose(2, 1).contiguous().view(B*N, -1, D//N)
                                for x in (query, key, value))
        if mask is not None:
            mask = mask[:, None, :].expand(B, N, Tk).contiguous().view(B*N, -1)
        outputs = self.attention(query, key, value, mask, probs, beta, tau, weights)  # (B x N) x T x (D/N)
        outputs = outputs.contiguous().view(B, N, -1, D//N).transpose(2, 1).contiguous().view(B, -1, D)

        if feedback is not None:
            feedback.append(probs[0].view(B, N, Tq, Tk))

        if self.use_wo:
            return self.wo(outputs)
        return outputs

class NonresidualBlock(nn.Module):

    def __init__(self, layer, d_model, d_hidden, drop_ratio, pos=0):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)
        self.pos = pos

    def forward(self, *x):
        return self.layernorm(self.dropout(self.layer(*x)))


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, d_hidden, drop_ratio, pos=0):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)
        self.pos = pos

    def forward(self, *x):
        return self.layernorm(x[self.pos] + self.dropout(self.layer(*x)))

class HighwayBlock(nn.Module):

    def __init__(self, layer, d_model, d_hidden, drop_ratio, pos=0):
        super().__init__()
        self.layer = layer
        self.gate = FeedForward(d_model, d_hidden)
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)
        self.pos = pos

    def forward(self, *x):
        g = F.sigmoid(self.gate(x[self.pos]))
        return self.layernorm(x[self.pos] * g + self.dropout(self.layer(*x)) * (1 - g))

class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = Linear(d_model, d_hidden)
        self.linear2 = Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead2(args.d_model, args.d_model, args.n_heads,
                       args.drop_ratio, use_wo=args.use_wo),
            args.d_model, args.d_hidden, args.drop_ratio)
        self.feedforward = args.block_cls(
            FeedForward(args.d_model, args.d_hidden),
            args.d_model, args.d_hidden, args.drop_ratio )

    def forward(self, x, mask=None):
        x = self.selfattn(x, x, x, mask)
        x = self.feedforward(x)
        return x

class DecoderLayer(nn.Module):

    def __init__(self, args, causal=True, diag=False,
                positional=False):
        super().__init__()

        self.positional = positional
        self.selfattn = ResidualBlock(
            MultiHead2(args.d_model, args.d_model, args.n_heads,
                    args.drop_ratio, causal=causal, diag=diag,
                    use_wo=args.use_wo),
            args.d_model, args.d_hidden, args.drop_ratio)

        self.attention = ResidualBlock(
            MultiHead2(args.d_model, args.d_model, args.n_heads,
                    args.drop_ratio, use_wo=args.use_wo),
            args.d_model, args.d_hidden, args.drop_ratio)

        if positional:
            self.pos_selfattn = ResidualBlock(
            MultiHead2(args.d_model, args.d_model, args.n_heads,
                    args.drop_ratio, causal=causal, diag=diag,
                    use_wo=args.use_wo),
            args.d_model, args.d_hidden, args.drop_ratio, pos=2)

        self.feedforward = args.block_cls(
            FeedForward(args.d_model, args.d_hidden),
            args.d_model, args.d_hidden, args.drop_ratio )

    def forward(self, x, encoding, p=None, mask_src=None, mask_trg=None, feedback=None):

        feedback_src = []
        feedback_trg = []

        x = self.selfattn(x, x, x, mask_trg, feedback_trg)   #

        if self.positional:
            pos_encoding, weights = positional_encodings_like(x), None
            x = self.pos_selfattn(pos_encoding, pos_encoding, x, mask_trg, None, weights)  # positional attention

        x = self.attention(x, encoding, encoding, mask_src, feedback_src)

        x = self.feedforward(x)

        if feedback is not None:
            if 'source' not in feedback:
                feedback['source'] = feedback_src
            else:
                feedback['source'] += feedback_src

            if 'target' not in feedback:
                feedback['target'] = feedback_trg
            else:
                feedback['target'] += feedback_trg
        return x

class Encoder(nn.Module):

    def __init__(self, field, args):
        super().__init__()

        if args.dataset != "mscoco":
            if args.share_embed:
                self.out = Linear(args.d_model, len(field.vocab), bias=False)
            else:
                self.embed = nn.Embedding(len(field.vocab), args.d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(args) for i in range(args.n_layers)])
        self.dropout = nn.Dropout(args.input_drop_ratio)
        if args.dataset != "mscoco":
            self.field = field
        self.d_model = args.d_model
        self.share_embed = args.share_embed
        self.dataset = args.dataset

    def forward(self, x, mask=None):
        if self.dataset != "mscoco":
            if self.share_embed:
                x = F.embedding(x, self.out.weight * math.sqrt(self.d_model))
            else:
                x = self.embed(x)
        x += positional_encodings_like(x)
        encoding = [x]

        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
            encoding.append(x)
        return encoding

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class EncoderCNN(nn.Module):
    def __init__(self, args):
        super(EncoderCNN, self).__init__()

        self.d_encoder = 512 # hardcoded because of resnet 512 hidden size
        self.d_model = args.d_model
        if self.d_encoder != self.d_model:
            self.conv = conv3x3(self.d_encoder, self.d_model, stride=1)
            self.bn = nn.BatchNorm2d(self.d_model)

    def forward(self, features):
        if self.d_encoder != self.d_model:
            return self.bn(self.conv(features))
        else:
            return features

class Decoder(nn.Module):

    def __init__(self, field, args, causal=True, positional=False, diag=False, out=None):

        super().__init__()

        self.layers = nn.ModuleList(
            [DecoderLayer(args, causal, diag, positional)
            for i in range(args.n_layers)])

        if out is None:
            self.out = Linear(args.d_model, len(field.vocab), bias=False, out_norm=args.out_norm)
        else:
            self.out = out

        self.dropout = nn.Dropout(args.input_drop_ratio)
        self.out_norm = args.out_norm
        self.d_model = args.d_model
        self.field = field
        self.length_ratio = args.length_ratio
        self.positional = positional
        self.enc_last = args.enc_last
        self.dataset = args.dataset
        self.length_dec = args.length_dec

    def forward(self, x, encoding, source_masks=None, decoder_masks=None,
                input_embeddings=False, positions=None, feedback=None):
        # x : decoder_inputs

        if self.out_norm:
            out_weight = self.out.weight / (1e-6 + torch.sqrt((self.out.weight ** 2).sum(0, keepdim=True)))
        else:
            out_weight = self.out.weight

        if not input_embeddings:  # NOTE only for Transformer
            if x.ndimension() == 2:
                x = F.embedding(x, out_weight * math.sqrt(self.d_model))
            elif x.ndimension() == 3:  # softmax relaxiation
                x = x @ out_weight * math.sqrt(self.d_model)  # batch x len x embed_size

        x += positional_encodings_like(x)
        x = self.dropout(x)

        if self.enc_last:
            for l, layer in enumerate(self.layers):
                x = layer(x, encoding[-1], mask_src=source_masks, mask_trg=decoder_masks, feedback=feedback)
        else:
            for l, (layer, enc) in enumerate(zip(self.layers, encoding[1:])):
                x = layer(x, enc, mask_src=source_masks, mask_trg=decoder_masks, feedback=feedback)
        return x

    def greedy(self, encoding, mask_src=None, mask_trg=None, feedback=None):

        encoding = encoding[1:]
        B, T, C = encoding[0].size()  # batch-size, decoding-length, size
        if self.dataset == "mscoco":
            T = self.length_dec
        else:
            T *= self.length_ratio
        T = int(T)

        outs = Variable(encoding[0].data.new(B, T + 1).long().fill_(
                    self.field.vocab.stoi['<init>']))
        hiddens = [Variable(encoding[0].data.new(B, T, C).zero_())
                    for l in range(len(self.layers) + 1)]
        embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])

        eos_yet = encoding[0].data.new(B).byte().zero_()

        attentions = []

        for t in range(T):
            #torch.cuda.nvtx.mark(f'greedy:{t}')
            torch.cuda.nvtx.mark('greedy:{}'.format(t))
            hiddens[0][:, t] = self.dropout(
                hiddens[0][:, t] + F.embedding(outs[:, t], embedW))

            inter_attention = []
            for l in range(len(self.layers)):
                x = hiddens[l][:, :t+1]
                x = self.layers[l].selfattn(hiddens[l][:, t:t+1], x, x)   # we need to make the dimension 3D
                hiddens[l + 1][:, t] = self.layers[l].feedforward(
                    self.layers[l].attention(x, encoding[l], encoding[l], mask_src, inter_attention))[:, 0]

            inter_attention = torch.cat(inter_attention, 1)
            attentions.append(inter_attention)

            _, preds = self.out(hiddens[-1][:, t]).max(-1)
            preds[eos_yet] = self.field.vocab.stoi['<pad>']

            eos_yet = eos_yet | (preds.data == self.field.vocab.stoi['<eos>'])
            outs[:, t + 1] = preds
            if eos_yet.all():
                break

        if feedback is not None:
            feedback['source'] = torch.cat(attentions, 2)

        return outs[:, 1:t+2]

    def beam_search(self, encoding, mask_src=None, mask_trg=None, width=2, alpha=0.6):  # width: beamsize, alpha: length-norm
        encoding = encoding[1:]
        W = width
        B, T, C = encoding[0].size()

        # expanding
        for i in range(len(encoding)):
            encoding[i] = encoding[i][:, None, :].expand(
                B, W, T, C).contiguous().view(B * W, T, C)
        mask_src = mask_src[:, None, :].expand(B, W, T).contiguous().view(B * W, T)

        T *= self.length_ratio
        outs = Variable(encoding[0].data.new(B, W, T + 1).long().fill_(
            self.field.vocab.stoi['<init>']))

        logps = Variable(encoding[0].data.new(B, W).float().fill_(0))  # scores
        hiddens = [Variable(encoding[0].data.new(B, W, T, C).zero_())  # decoder states: batch x beamsize x len x h
                    for l in range(len(self.layers) + 1)]
        embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        eos_yet = encoding[0].data.new(B, W).byte().zero_()  # batch x beamsize, all the sentences are not finished yet.
        eos_mask = eos_yet.float().fill_(-INF)[:, :, None].expand(B, W, W)
        eos_mask[:, :, 0] = 0  # batch x beam x beam

        for t in range(T):
            hiddens[0][:, :, t] = self.dropout(
                hiddens[0][:, :, t] + F.embedding(outs[:, :, t], embedW))
            for l in range(len(self.layers)):
                x = hiddens[l][:, :, :t + 1].contiguous().view(B * W, -1, C)
                x = self.layers[l].selfattn(x[:, -1:, :], x, x)
                hiddens[l + 1][:, :, t] = self.layers[l].feedforward(
                    self.layers[l].attention(x, encoding[l], encoding[l], mask_src)).view(
                        B, W, C)

            # topk2_logps: scores, topk2_inds: top word index at each beam, batch x beam x beam
            topk2_logps, topk2_inds = log_softmax(
                self.out(hiddens[-1][:, :, t])).topk(W, dim=-1)

            # mask out the sentences which are finished
            topk2_logps = topk2_logps * Variable(eos_yet[:, :, None].float() * eos_mask + 1 - eos_yet[:, :, None].float())
            topk2_logps = topk2_logps + logps[:, :, None]

            if t == 0:
                logps, topk_inds = topk2_logps[:, 0].topk(W, dim=-1)
            else:
                logps, topk_inds = topk2_logps.view(B, W * W).topk(W, dim=-1)

            topk_beam_inds = topk_inds.div(W)
            topk_token_inds = topk2_inds.view(B, W * W).gather(1, topk_inds)
            eos_yet = eos_yet.gather(1, topk_beam_inds.data)

            logps = logps * (1 - Variable(eos_yet.float()) * 1 / (t + 2)).pow(alpha)
            outs = outs.gather(1, topk_beam_inds[:, :, None].expand_as(outs))
            outs[:, :, t + 1] = topk_token_inds
            topk_beam_inds = topk_beam_inds[:, :, None, None].expand_as(
                hiddens[0])
            for i in range(len(hiddens)):
                hiddens[i] = hiddens[i].gather(1, topk_beam_inds)
            eos_yet = eos_yet | (topk_token_inds.data == self.field.vocab.stoi['<eos>'])
            if eos_yet.all():
                return outs[:, 0, 1:]
        return outs[:, 0, 1:]

class MultiGPUTransformer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sources, source_masks, decoder_inputs, decoder_masks, targets, target_masks):
        # get encoding first
        encoding = self.model.encoding(sources, source_masks)
        # rest
        out = self.model(encoding, source_masks, decoder_inputs, decoder_masks)
        loss = self.model.cost(targets, target_masks, out=out)
        loss = loss.view(-1)
        return loss

class MultiGPUFastTransformer(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

    def forward(self, sources, source_masks, attention, decoder_masks, targets, target_masks, denoise, denoise_cor, trg_len_option="reference"):
        # get encoding first
        encoding = self.model.encoding(sources, source_masks)

        # if trg_len_option is predict then given encoding predict target len offset
        if trg_len_option == "predict":
            target_offset = (target_masks.sum(-1) - source_masks.sum(-1)).clamp(min=-self.args.max_offset, max=self.args.max_offset)
            source_len = source_masks.sum(-1)

            pred_target_len_inputs_ = 0
            for i_layer_ in range(self.args.n_layers):
                pred_target_len_inputs_ += encoding[i_layer_]
            pred_target_len_inputs_ = pred_target_len_inputs_.detach()

            pred_target_offset_logits = self.model.pred_len(pred_target_len_inputs_.mean(1))
            pred_target_offset_logits = self.model.pred_len_drop( pred_target_offset_logits )
            # make sure that min targets is 0
            pred_target_len_loss = F.cross_entropy(pred_target_offset_logits, (target_offset + self.args.max_offset).long(), reduce=False)
            # calculate pred target len
            pred_target_offset = pred_target_offset_logits.max(-1)[1] - self.args.max_offset
            pred_target_len = source_len.long() + pred_target_offset

        batch_size = encoding[0].size(0)
        d_model = encoding[0].size(2)

        # given decoder_masks and encoding[0] (embedding) gather correct input
        decoder_inputs = torch.gather(encoding[0], dim=1, index=attention)

        # now do decoder forward pass
        losses = []
        for iter_ in range(self.args.train_repeat_dec):
            curr_iter = min(iter_, self.args.num_decs-1)
            next_iter = min(curr_iter + 1, self.args.num_decs-1)

            out = self.model(encoding, source_masks, decoder_inputs, decoder_masks, iter_=curr_iter, return_probs=False)

            loss = self.model.cost2(targets, target_masks, out=out, iter_=curr_iter)

            if not denoise[iter_]:
                logits = self.model.decoder[curr_iter].out(out)
                if self.args.use_argmax:
                    _, argmax = torch.max(logits, dim=-1)
                else:
                    probs = softmax(logits)
                    probs_sz = probs.size()
                    argmax = torch.multinomial(probs.detach().contiguous().view(-1, probs_sz[-1]), 1).view(*probs_sz[:-1])

            losses.append(loss)

            # set inputs for decoder at the next iteration
            decoder_inputs_ = 0
            denoising_mask = 1
            if self.args.next_dec_input in ["both", "emb"]:
                if denoise[iter_]:
                    decoder_inputs_ += F.embedding(denoise_cor[iter_], self.model.decoder[next_iter].out.weight * math.sqrt(self.args.d_model))
                else:
                    decoder_inputs_ += F.embedding(argmax, self.model.decoder[next_iter].out.weight * math.sqrt(self.args.d_model))
            if self.args.next_dec_input in ["both", "out"]:
                decoder_inputs_ += out
            decoder_inputs = decoder_inputs_

        if trg_len_option == "predict":
            return losses, pred_target_len_loss, pred_target_len
        else:
            return losses


class Transformer(nn.Module):

    def __init__(self, src=None, trg=None, args=None):
        super().__init__()
        if args.dataset != "mscoco":
            self.is_mscoco = False
            # prepare regular translation encoder and decoder
            self.encoder = Encoder(src, args)
            self.decoder = Decoder(trg, args)
            self.field = trg
            self.share_embed = args.share_embed
            if args.share_embed:
                self.encoder.out.weight = self.decoder.out.weight
        else:
            # prepare image encoder and decoder
            self.is_mscoco = True
            mscoco_dataset = trg
            #self.encoder = EncoderCNN(args)
            self.encoder = Encoder(src, args)
            self.decoder = Decoder(mscoco_dataset, args)
            self.field = mscoco_dataset
            self.share_embed = False

        self.n_layers = args.n_layers
        self.d_model = args.d_model

    def denum(self, data, target=True):
        field = self.decoder.field if target else self.encoder.field
        return field.reverse(data.unsqueeze(0))[0]

    def apply_mask(self, inputs, mask, p=1):
        _mask = Variable(mask.long())
        #outputs = inputs * _mask + (1 - _mask) * p
        outputs = inputs * _mask + (torch.mul(_mask, -1) + 1 ) * p
        return outputs

    def apply_mask_cost(self, loss, mask, batched=False):
        loss.data *= mask
        cost = loss.sum() / (mask.sum() + TINY)

        if not batched:
            return cost

        loss = loss.sum(1, keepdim=True) / (TINY + Variable(mask).sum(1, keepdim=True))
        return cost, loss

    def output_decoding(self, outputs, unbpe=True):
        field, text = outputs
        if field is 'src':
            return self.encoder.field.reverse(text.data, unbpe)
        else:
            return self.decoder.field.reverse(text.data, unbpe)

    def prepare_sources(self, batch, masks=None):
        masks = self.prepare_masks(batch.src) if masks is None else masks
        return batch.src, masks

    def encoding(self, encoder_inputs, source_masks=None):
        if self.is_mscoco:
            return self.encoder(encoder_inputs)
        else:
            return self.encoder(encoder_inputs, source_masks)

    def prepare_targets(self, batch, targets=None, masks=None):
        if targets is None:
            targets = batch.trg[:, 1:].contiguous()
        masks = self.prepare_masks(targets) if masks is None else masks
        return targets, masks

    def prepare_decoder_inputs(self, trg_inputs, inputs=None, masks=None, bp=1.00):
        decoder_inputs = trg_inputs[:, :-1].contiguous()
        decoder_masks = self.prepare_masks(trg_inputs[:, 1:], bp=bp) if masks is None else masks
        # NOTE why [1:], not [:-1]?

        return decoder_inputs, decoder_masks

    def change_bp_masks(self, masks, bp):
        input_lengths = np.int32( masks.sum(1).cpu().numpy() )
        batch_size, seq_len = masks.size()
        add_pad = [ int( math.floor( each_len * ( (1 / bp) - 1.0 ) ) ) for each_len in input_lengths]
        if max(add_pad) > 0 :
            add_mask = torch.zeros((batch_size, max(add_pad))).float() # NOTE we add masks of ones at the front!
            if masks.is_cuda:
                add_mask = add_mask.cuda(masks.get_device())
            masks = torch.cat((masks, add_mask), dim=1)
            for bidx in range(input_lengths.shape[0]):
                if add_pad[bidx] > 0:
                    masks[bidx, input_lengths[bidx]:input_lengths[bidx]+add_pad[bidx]] = 1
        return masks

    def prepare_masks(self, inputs, bp=1.0):
        if inputs.ndimension() == 2:
            masks = (inputs.data != self.field.vocab.stoi['<pad>']).float()
        else: # NOTE FALSE
            masks = (inputs.data[:, :, self.field.vocab.stoi['<pad>']] != 1).float()

        if bp < 1.0:
            masks = self.change_bp_masks(masks, bp)

        return masks

    def find_captions_length(self, all_captions):
        # find length of each caption
        all_captions_lengths = []
        # list of lists
        if type(all_captions[0]) == list:
            num_captions = len(all_captions[0])
            for i in range(num_captions):
                caption_length = 0
                for j in range(len(all_captions)):
                    caption_length += len(all_captions[j][i].split(' '))
                caption_length = int(caption_length / len(all_captions))
                all_captions_lengths.append(caption_length)
        else:
            for cap in all_captions:
                all_captions_lengths.append(len(cap.split(' ')))

        return all_captions_lengths

    def quick_prepare_mscoco(self, batch, all_captions=None, fast=True, inputs_dec='pool', trg_len_option=None, max_len=20, trg_len_dic=None, decoder_inputs=None, targets=None, decoder_masks=None, target_masks=None, source_masks=None, bp=1.00, gpu=True):
        features_beforepool, captions = batch[0], batch[1]
        batch_size, d_model = features_beforepool.size(0), features_beforepool.size(1)

        # batch_size x 49 x 512
        features_beforepool = features_beforepool.view(batch_size, d_model, 49).transpose(1, 2)
        if gpu:
            encoding = self.encoding(Variable(features_beforepool, requires_grad=False).cuda(), source_masks) # batch of resnet features
            source_masks = torch.FloatTensor(batch_size, 49).fill_(1).cuda()
            targets = self.prepare_target_captions(captions, self.field.vocab.stoi).cuda()
        else:
            encoding = self.encoding(Variable(features_beforepool, requires_grad=False), source_masks) # batch of resnet features
            source_masks = torch.FloatTensor(batch_size, 49).fill_(1)
            targets = self.prepare_target_captions(captions, self.field.vocab.stoi)

        # list of batch_size
        all_captions_lengths = self.find_captions_length(all_captions)

        # predicted decoder lens
        if trg_len_option == "predict":
            # batch_size tensor
            if gpu:
                target_len = Variable(torch.from_numpy(np.clip(np.array(all_captions_lengths), 0, self.max_offset)).cuda(), requires_grad=False)
            else:
                target_len = Variable(torch.from_numpy(np.clip(np.array(all_captions_lengths), 0, self.max_offset)), requires_grad=False)

            # HARDCODED (4 layer model) !!!
            pred_target_len_logits = self.pred_len((encoding[0]+encoding[1]+encoding[2]+encoding[3]+encoding[4]).mean(1))
            pred_target_len_loss = F.cross_entropy(pred_target_len_logits, target_len.long())
            pred_target_len = pred_target_len_logits.max(-1)[1]

        if fast == False:
            decoder_inputs, decoder_masks   = self.prepare_decoder_inputs(targets, decoder_inputs, decoder_masks)     # prepare decoder-inputs
        else:
            if trg_len_option == "fixed":
                decoder_len = int(max_len)
                decoder_masks = torch.ones(batch_size, decoder_len)
                if gpu:
                    decoder_masks = decoder_masks.cuda(encoding[0].get_device())

                # TODO ADD BP OPTION
            elif trg_len_option == "reference" or (trg_len_option == "predict" and self.use_predicted_trg_len == False):
                decoder_len = max(all_captions_lengths)
                decoder_masks = np.zeros((batch_size, decoder_len))
                for idx in range(decoder_masks.shape[0]):
                    decoder_masks[idx][:all_captions_lengths[idx]] = 1
                decoder_masks = torch.from_numpy(decoder_masks).float()
                if gpu:
                    decoder_masks = decoder_masks.cuda(encoding[0].get_device())

            if trg_len_option == "predict":
                if self.use_predicted_trg_len:
                    pred_target_len = pred_target_len.data.cpu().numpy()
                    decoder_len = np.max(pred_target_len)
                    decoder_masks = np.zeros((batch_size, decoder_len))
                    for idx in range(pred_target_len.shape[0]):
                        decoder_masks[idx][:pred_target_len[idx]] = 1
                    decoder_masks = torch.from_numpy(decoder_masks).float()
                    if gpu:
                        decoder_masks = decoder_masks.cuda(encoding[0].get_device())
                    if bp < 1.0:
                        decoder_masks = self.change_bp_masks(decoder_masks, bp)

                if not self.use_predicted_trg_len:
                    pred_target_len = pred_target_len.data.cpu().numpy()

                target_len = target_len.data.cpu().numpy()

                # calculate error for predicted target length
                pred_target_len_correct = np.sum(pred_target_len == target_len)*100/batch_size
                pred_target_len_approx = np.sum(np.abs(pred_target_len - target_len) < 5)*100/batch_size
                average_target_len_correct = 0
                average_target_len_approx = 0

                rest = [pred_target_len_loss, pred_target_len_correct, pred_target_len_approx, average_target_len_correct, average_target_len_approx]

            if inputs_dec == 'pool':
                # batch_size x 1 x 512
                decoder_inputs = torch.mean(features_beforepool, 1, keepdim=True)
                decoder_inputs = decoder_inputs.repeat(1, int(decoder_len), 1)
                decoder_inputs = Variable(decoder_inputs, requires_grad=False)
                if gpu:
                    decoder_inputs = decoder_inputs.cuda(encoding[0].get_device())
            elif inputs_dec == 'zeros':
                decoder_inputs = Variable(torch.zeros(batch_size, int(decoder_len), d_model), requires_grad=False)
                if gpu:
                    decoder_inputs = decoder_inputs.cuda(encoding[0].get_device())

        # REMOVE THE FIRST <INIT> TAG FROM CAPTIONS
        targets = targets[:, 1:]
        if gpu:
            target_masks = (targets != 1).float().cuda().data
        else:
            target_masks = (targets != 1).float().data

        if trg_len_option != "predict":
            rest = []
        sources = None

        return decoder_inputs, decoder_masks, targets, target_masks, sources, source_masks, encoding, decoder_inputs.size(0), rest

    def prepare_target_captions(self, captions, vocab):
        # captions : batch_size X seq_len
        lst = []
        batch_size = len(captions)
        for bidx in range(batch_size):
            lst.append( ["<init>"] + captions[ bidx ].lower().split() + ["<eos>"] )
            #lst.append( [ vocab[idx] for idx in captions[ random.randint(0,4) ][ bidx ].lower().split() ] )
        lst = [[vocab[idx] if idx in vocab else 0 for idx in sentence] for sentence in lst]
        seq_len = max( [len(xx) for xx in lst] )
        captions = np.ones((batch_size, seq_len))
        for bidx in range(batch_size):
            min_len = min(seq_len, len(lst[bidx]))
            captions[bidx, :min_len] = np.array(lst[bidx][:min_len])
        captions = torch.from_numpy(captions).long()
        return Variable(captions, requires_grad=False)

    def quick_prepare_window(self, batch, fast=True, trg_len_option=None, trg_len_ratio=2.0, trg_len_dic=None, decoder_inputs=None, targets=None, decoder_masks=None, target_masks=None, source_masks=None, bp=1.00, window=5):
        sources,        source_masks    = self.prepare_sources(batch, source_masks)
        encoding                        = self.encoding(sources, source_masks)
        targets,        target_masks    = self.prepare_targets(batch, targets, decoder_masks)  # prepare decoder-targets

        if trg_len_option == "predict":
            target_offset = Variable((target_masks.sum(-1) - source_masks.sum(-1)).clamp_(-self.max_offset, self.max_offset), requires_grad=False) # batch_size tensor
            source_len = Variable(source_masks.sum(-1), requires_grad=False)

            pred_target_len_inputs_ = 0
            for i_layer_ in range(self.n_layers):
                pred_target_len_inputs_ += encoding[i_layer_]
            pred_target_len_inputs_ = pred_target_len_inputs_.detach()

            pred_target_offset_logits = self.pred_len(pred_target_len_inputs_.mean(1))
            pred_target_offset_logits = self.pred_len_drop( pred_target_offset_logits )
            pred_target_len_loss = F.cross_entropy(pred_target_offset_logits, (target_offset + self.max_offset).long())
            pred_target_offset = pred_target_offset_logits.max(-1)[1] - self.max_offset
            pred_target_len = source_len.long() + pred_target_offset

        d_model = encoding[0].size(-1)
        batch_size, src_max_len = source_masks.size()
        rest = []

        if fast:
            if trg_len_option == "predict":
                # convert to numpy arrays first
                source_len = source_masks.sum(-1).cpu().numpy()
                target_len = target_masks.sum(-1).cpu().numpy()
                pred_target_len = pred_target_len.data.cpu().numpy()
                # make sure pred_target_len has no negative numbers
                pred_target_len[pred_target_len<=2] = 2

                # repeat pred_target_len window amount of times
                pred_target_len_corr = np.zeros((batch_size * window))
                for w in range(window):
                    pred_target_len_corr[w*batch_size:(w+1)*batch_size] = pred_target_len + (w - window//2)

                pred_target_len = np.int32(pred_target_len_corr[:])
                # make sure pred_target_len has no negative numbers
                pred_target_len[pred_target_len<=2] = 2

                # create decoder masks based on predicted len
                pred_decoder_max_len = max(pred_target_len)
                pred_decoder_masks = np.zeros((pred_target_len.shape[0], pred_decoder_max_len))
                for idx in range(pred_target_len.shape[0]):
                    pred_decoder_masks[idx][:pred_target_len[idx]] = 1
                pred_decoder_masks = torch.from_numpy(pred_decoder_masks).float()
                if source_masks.is_cuda:
                    pred_decoder_masks = pred_decoder_masks.cuda()
                if bp < 1.0:
                    pred_decoder_masks = self.change_bp_masks(pred_decoder_masks, bp)

            encoding = [enc.repeat(window, 1, 1) for enc in encoding]
            sources = sources.repeat(window, 1)
            source_masks = source_masks.repeat(window, 1)

            if trg_len_option == "predict":
                pred_decoder_inputs, pred_decoder_masks   = self.prepare_initial(encoding, sources, source_masks, pred_decoder_masks)

        rest = []
        if trg_len_option == "predict":
            return pred_decoder_inputs, pred_decoder_masks, targets, target_masks, sources, source_masks, encoding, batch_size, rest


    def quick_prepare(self, batch, fast=True, trg_len_option=None, trg_len_ratio=2.0, trg_len_dic=None, decoder_inputs=None, targets=None, decoder_masks=None, target_masks=None, source_masks=None, bp=1.00):
        sources,        source_masks    = self.prepare_sources(batch, source_masks)
        encoding                        = self.encoding(sources, source_masks)
        targets,        target_masks    = self.prepare_targets(batch, targets, decoder_masks)  # prepare decoder-targets

        if trg_len_option == "predict":
            target_offset = Variable((target_masks.sum(-1) - source_masks.sum(-1)).clamp_(-self.max_offset, self.max_offset), requires_grad=False) # batch_size tensor
            source_len = Variable(source_masks.sum(-1), requires_grad=False)

            pred_target_len_inputs_ = 0
            for i_layer_ in range(self.n_layers):
                pred_target_len_inputs_ += encoding[i_layer_]
            pred_target_len_inputs_ = pred_target_len_inputs_.detach()

            pred_target_offset_logits = self.pred_len(pred_target_len_inputs_.mean(1))
            pred_target_offset_logits = self.pred_len_drop( pred_target_offset_logits )
            pred_target_len_loss = F.cross_entropy(pred_target_offset_logits, (target_offset + self.max_offset).long())
            pred_target_offset = pred_target_offset_logits.max(-1)[1] - self.max_offset
            pred_target_len = source_len.long() + pred_target_offset

        d_model = encoding[0].size(-1)
        batch_size, src_max_len = source_masks.size()
        rest = []

        if fast:
            # compute decoder_masks (for reference case)
            _, decoder_masks   = self.prepare_decoder_inputs(batch.trg, decoder_inputs, decoder_masks, bp=bp)

            if trg_len_option == "predict":
                # convert to numpy arrays first
                source_len = source_masks.sum(-1).cpu().numpy()
                target_len = target_masks.sum(-1).cpu().numpy()
                pred_target_len = pred_target_len.data.cpu().numpy()
                # make sure pred_target_len has no negative numbers
                pred_target_len[pred_target_len<=2] = 2

                # create decoder masks based on predicted len
                pred_decoder_max_len = max(pred_target_len)
                pred_decoder_masks = np.zeros((batch_size, pred_decoder_max_len))
                for idx in range(pred_target_len.shape[0]):
                    pred_decoder_masks[idx][:pred_target_len[idx]] = 1
                pred_decoder_masks = torch.from_numpy(pred_decoder_masks).float()
                if source_masks.is_cuda:
                    pred_decoder_masks = pred_decoder_masks.cuda()
                if bp < 1.0:
                    pred_decoder_masks = self.change_bp_masks(pred_decoder_masks, bp)

                # check the results of predicting target length
                pred_target_len_correct = np.sum(pred_target_len == target_len)*100/batch_size
                pred_target_len_approx = np.sum(np.abs(pred_target_len - target_len) < 5)*100/batch_size

                if trg_len_dic != None:
                    # results with average len
                    average_target_len = [query_trg_len_dic(trg_len_dic, source) for source in source_len]
                    average_target_len = np.array(average_target_len)
                    average_target_len_correct = np.sum(average_target_len == target_len)*100/batch_size
                    average_target_len_approx = np.sum(np.abs(average_target_len - target_len) < 5)*100/batch_size
                else:
                    average_target_len_correct = 0.0
                    average_target_len_approx = 0.0

                rest = [pred_target_len_loss, pred_target_len_correct, pred_target_len_approx, average_target_len_correct, average_target_len_approx]

            decoder_inputs, decoder_masks   = self.prepare_initial(encoding, sources, source_masks, decoder_masks)
            if trg_len_option == "predict":
                pred_decoder_inputs, pred_decoder_masks   = self.prepare_initial(encoding, sources, source_masks, pred_decoder_masks)

        else:
            decoder_inputs, decoder_masks   = self.prepare_decoder_inputs(batch.trg, decoder_inputs, decoder_masks)     # prepare decoder-inputs

        if trg_len_option == "predict":
            return decoder_inputs, decoder_masks, pred_decoder_inputs, pred_decoder_masks, targets, target_masks, sources, source_masks, encoding, decoder_inputs.size(0), rest
        else:
            return decoder_inputs, decoder_masks, targets, target_masks, sources, source_masks, encoding, decoder_inputs.size(0), rest

    def forward(self, encoding, source_masks, decoder_inputs, decoder_masks,
                decoding=False, beam=1, alpha=0.6, return_probs=False, positions=None, feedback=None):

        if (return_probs and decoding) or (not decoding):
            out = self.decoder(decoder_inputs, encoding, source_masks, decoder_masks)

        if decoding:
            if beam == 1:  # greedy decoding
                output = self.decoder.greedy(encoding, source_masks, decoder_masks, feedback=feedback)
            else:
                output = self.decoder.beam_search(encoding, source_masks, decoder_masks, beam, alpha)

            if return_probs:
                return output, out, self.decoder.out(out) # NOTE don't do softmax for validation
                #return output, out, softmax(self.decoder.out(out))
            return output

        if return_probs:
            return out, softmax(self.decoder.out(out))
        return out

    def cost(self, decoder_targets, decoder_masks, out=None):
        # get loss in a sequence-format to save computational time.
        decoder_targets, out = prepare_cost(decoder_targets, out, decoder_masks.byte())
        logits = self.decoder.out(out)
        loss = F.cross_entropy(logits, decoder_targets)
        return loss

    def batched_cost(self, decoder_targets, decoder_masks, probs, batched=False):
        # get loss in a batch-mode

        if decoder_targets.ndimension() == 2:  # batch x length
            loss = -torch.log(probs + TINY).gather(2, decoder_targets[:, :, None])[:, :, 0]  # batch x length
        else:
            loss = -(torch.log(probs + TINY) * decoder_targets).sum(-1)
        return self.apply_mask_cost(loss, decoder_masks, batched)

class FastTransformer(Transformer):

    def __init__(self, src=None, trg=None, args=None):
        super(Transformer, self).__init__()
        self.is_mscoco = args.dataset == "mscoco"
        self.decoder_input_how = args.decoder_input_how
        self.encoder = Encoder(src, args)
        '''
        if self.is_mscoco == False:
            self.encoder = Encoder(src, args)
        else:
            self.encoder = EncoderCNN(args)
        '''
        self.decoder = nn.ModuleList()
        for ni in range(args.num_decs):
            self.decoder.append(Decoder(trg, args,
                                    causal=False,
                                    positional=args.positional,
                                    diag=args.diag,
                                    out=self.encoder.out if args.share_embed_enc_dec1 and ni == 0 else None)
                                )
        self.field = trg
        if self.is_mscoco == False:
            self.share_embed = args.share_embed
        else:
            self.share_embed = False
        self.train_repeat_dec = args.train_repeat_dec
        self.num_decs = args.num_decs
        if args.trg_len_option == "predict":
            if args.dataset != "mscoco":
                self.pred_len = Linear(args.d_model, 2*args.max_offset + 1)
            else:
                self.pred_len = Linear(args.d_model, args.max_offset+1)
            self.pred_len_drop = nn.Dropout(args.drop_len_pred)
            self.max_offset = args.max_offset
            self.use_predicted_trg_len = args.use_predicted_trg_len
        self.n_layers = args.n_layers
        self.d_model = args.d_model

    def output_decoding(self, outputs):
        field, text = outputs
        if field is 'src':
            return self.encoder.field.reverse(text.data)
        else:
            return self.decoder[0].field.reverse(text.data)

    # decoder_masks already decided
    # computes decoder_inputs
    def prepare_initial(self, encoding, source=None, source_masks=None, decoder_masks=None,
                        N=1, tau=1):

        decoder_input_how = self.decoder_input_how
        d_model = encoding[0].size()[-1]
        attention = linear_attention(source_masks, decoder_masks, decoder_input_how)

        if decoder_input_how in ["copy", "pad", "wrap"]:
            attention = self.apply_mask(attention, decoder_masks, p=1) # p doesn't matter cos masked out
            attention = attention[:,:,None].expand(*attention.size(), d_model)
            decoder_inputs = torch.gather(encoding[0], dim=1, index=attention)

        elif decoder_input_how == "interpolate":
            decoder_inputs = matmul(attention, encoding[0]) # batch x max_trg x size

        return decoder_inputs, decoder_masks

    def forward(self, encoding, source_masks, decoder_inputs, decoder_masks,
                decoding=False, beam=1, alpha=0.6,
                return_probs=False, positions=None, feedback=None, iter_=0, T=1):

        thedecoder = self.decoder[iter_]

        out = thedecoder(decoder_inputs, encoding, source_masks, decoder_masks,
                            input_embeddings=True, positions=positions, feedback=feedback)
        # out : output from the (-1)-th DecoderLayer

        if not decoding: # NOTE training
            if not return_probs:
                return out
            return out, softmax(thedecoder.out(out), T=T) # probs

        logits = thedecoder.out(out)

        if beam == 1:
            output = self.apply_mask(logits.max(-1)[1], decoder_masks) # NOTE given mask, set non-mask to 1
        else:
            output, decoder_masks = topK_search(logits, decoder_masks, N=beam)
            output = self.apply_mask(output, decoder_masks)

        if not return_probs:
            return output
        else:
            #return output, out, logits # NOTE don't do softmax for validation
            return output, out, softmax(logits, T=T)

    def cost(self, targets, target_mask, out=None, iter_=0, return_logits=False):
        # get loss in a sequence-format to save computational time.
        targets, out = prepare_cost(targets, out, target_mask.byte())
        logits = self.decoder[iter_].out(out)
        loss = F.cross_entropy(logits, targets)
        if return_logits:
            return loss, logits
        return loss

    # cost2 is just a different implementation of cost
    def cost2(self, targets, target_mask, out=None, iter_=0, return_logits=False):
        # prepare targets
        targets_sz = targets.size()
        targets = targets.view(targets_sz[0]*targets_sz[1])
        # prepare logits
        logits = self.decoder[iter_].out(out)
        logits_sz = logits.size()
        logits = logits.view(logits_sz[0]*logits_sz[1], logits_sz[2])
        loss = F.cross_entropy(logits, targets, ignore_index=1, reduce=False)
        loss = loss.view(targets_sz)
        if return_logits:
            return loss, logits
        return loss


def mask(targets, out, input_mask=None, return_mask=False):
    if input_mask is None:
        input_mask = (targets != 1)
    out_mask = input_mask.unsqueeze(-1).expand_as(out)

    if return_mask:
        return targets[input_mask], out[out_mask].view(-1, out.size(-1)), the_mask
    return targets[input_mask], out[out_mask].view(-1, out.size(-1))

def prepare_cost(targets, out, target_mask=None, return_mask=None):
    # targets : batch_size, seq_len
    # out     : batch_size, seq_len, vocab_size
    # target_mask : batch_size, seq_len
    if target_mask is None:
        target_mask = (targets != 1)

    if targets.size(1) < out.size(1):
        out = out[:, :targets.size(1), :]
    elif targets.size(1) > out.size(1):
        targets = targets[:, :out.size(1)]
        target_mask = target_mask[:, :out.size(1)]

    out_mask = target_mask.unsqueeze(-1).expand_as(out)

    if return_mask:
        return targets[target_mask], out[out_mask].view(-1, out.size(-1)), out_mask
    else:
        return targets[target_mask], out[out_mask].view(-1, out.size(-1))

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

        index_s = steps[:,None] * index_s[None,:] # batch_size X max_trg_len
        index_s = Variable(torch.round(index_s), requires_grad=False).long()
        return index_s

    elif decoder_input_how == "wrap":
        batch_size, max_src_len = source_masks.size()
        max_trg_len = decoder_masks.size(1)

        src_lens = source_masks.sum(-1).int()  # batch_size

        index_s = torch.arange(max_trg_len)[None,:]  # max_trg_len
        index_s = index_s.repeat(batch_size, 1) # (batch_size, max_trg_len)

        for sin in range(batch_size):
            if src_lens[sin]+1 < max_trg_len:
                index_s[sin, src_lens[sin]:2*src_lens[sin]] = index_s[sin, :src_lens[sin]]

        if decoder_masks.is_cuda:
            index_s = index_s.cuda(decoder_masks.get_device())

        return Variable(index_s, requires_grad=False).long()

    elif decoder_input_how == "pad":
        batch_size, max_src_len = source_masks.size()
        max_trg_len = decoder_masks.size(1)

        src_lens = source_masks.sum(-1).int() - 1  # batch_size

        index_s = torch.arange(max_trg_len)[None,:]  # max_trg_len
        index_s = index_s.repeat(batch_size, 1) # (batch_size, max_trg_len)

        for sin in range(batch_size):
            if src_lens[sin]+1 < max_trg_len:
                index_s[sin, src_lens[sin]+1:] = index_s[sin, src_lens[sin]]

        if decoder_masks.is_cuda:
            index_s = index_s.cuda(decoder_masks.get_device())

        return Variable(index_s, requires_grad=False).long()

    elif decoder_input_how == "interpolate":
        max_src_len = source_masks.size(1)
        max_trg_len = decoder_masks.size(1)
        src_lens = source_masks.sum(-1).float()  # batchsize
        trg_lens = decoder_masks.sum(-1).float()  # batchsize
        steps = src_lens / trg_lens          # batchsize
        index_t = torch.arange(0, max_trg_len)  # max_trg_len
        if decoder_masks.is_cuda:
            index_t = index_t.cuda(decoder_masks.get_device())
        index_t = steps[:, None] @ index_t[None, :]  # batch x max_trg_len
        index_s = torch.arange(0, max_src_len)  # max_src_len
        if decoder_masks.is_cuda:
            index_s = index_s.cuda(decoder_masks.get_device())
        indexxx_ = (index_s[None, None, :] - index_t[:, :, None]) ** 2  # batch x max_trg x max_src
        indexxx = softmax(Variable(-indexxx_.float() / 0.3 - INF * (1 - source_masks[:, None, :].float() )))  # batch x max_trg x max_src
        return indexxx
