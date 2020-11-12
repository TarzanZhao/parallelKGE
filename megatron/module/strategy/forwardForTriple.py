#coding:utf-8

import torch

from .MegatronStrategyModule import MegatronStrategyModule

class ForwardForTriple(MegatronStrategyModule):
    
    def __init__(self, model):
        super(ForwardForTriple, self).__init__(model)

    def forward(self, h = None, r = None, t = None):
        if h.dim() == r.dim() and r.dim() == t.dim() and t.dim() <= 2:
            # check the shape of tensors: h - [B, N] r - [B, N] t - [B, N] or h - [B] r - [B] t - [B] 
            assert self._compare_tensor_shape(h, r) and self._compare_tensor_shape(t, r), 'The shapes of input tensors cannot match with each other!'
            # the shape of score is [B, N] or [B]
            score = self.model.forward_for_triple(h, r, t)
        else:
            # the shape is invalid
            raise Exception('The shapes of input tensors are invalid!')