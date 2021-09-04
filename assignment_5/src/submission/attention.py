import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        # (B x T x C) is of dimension (batch x block_size x n_embd) which is (batch x l x d) in the handout.
        # nh should be number_of_heads, and hs would then stand for n_embed (or "dimensionality" d in the handout) per head

        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B - batch size, nh - num heads, T - blocksize, hs - hidden size)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # => batch size * num heads * block size * block size
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class SynthesizerAttention(nn.Module):
    """
    A synthesizer multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # MLP Params
        self.w1 = nn.Linear(config.n_embd, config.n_embd)
        self.w2 = nn.Parameter(torch.zeros(config.n_embd // config.n_head,
            config.block_size-1))
        self.b2 = nn.Parameter(torch.zeros(config.block_size-1))
        # value projection
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in
        #     the input sequence
        self.register_buffer("mask", torch.tril(
            torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.block_size = config.block_size

        nn.init.uniform_(self.w2,-0.001,0.001)  # fills w2 with random weights

    def forward(self, x, layer_past=None):

        ### TODO:
        ### [part g]: Write your SynthesizerAttention below.
        ###   Do not modify __init__().
        ### Hints:
        ###   - Paste over the CausalSelfAttention above and modify it minimally.
        ###   - Consider especially the parameters self.w1, self.w2 and self.b2.
        ###       How do these map to the matrices in the handout?

        ### START CODE HERE

        # print(f"nh - heads: {self.n_head}, T - block size: {self.block_size}, C - n_embd: {self.n_embd}")

        # (B x T x C) is of dimension (batch x block_size x n_embd) which is (batch x l x d) in the handout.
        # nh should be number_of_heads, and hs would then stand for n_embed (or "dimensionality" d in the handout) per head

        B, T, C = x.size()

        # print(f"x dim: {x.shape}")

        k = self.w1(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # print(f"shape of k: {k.shape}")  # (Batch, num heads, block size, hidden states / num heads)

        relu = nn.ReLU()
        relu_k = relu(k)  # zeroes some values of k

        q = relu_k @ self.w2[:, :T]

        # print(f"shape of q: {q.shape}")

        att = q + self.b2[:T]

        # print(f"shape of att (after adding second bias): {att.shape}")

        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10)

        att = F.softmax(att, dim=-1)

        att = self.attn_drop(att)

        # print(f"shape of att (after softmax): {att.shape}")

        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # print(f"shape of XV: {v.shape}")

        y = att @ v

        # print(f"shape of y before shape altering: {y.shape}")  # batch size x num heads x block size x (embed size / heads)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # print(f"shape of y after shape altering: {y.shape}")  # batch size x block_size x n_embed

        # output projection
        y = self.resid_drop(self.proj(y))

        # print(f"shape of y after residual drop: {y.shape}")

        return y
        ### END CODE HERE

        raise NotImplementedError