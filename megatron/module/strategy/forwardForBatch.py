#coding:utf-8

import torch

from .MegatronStrategyModule import MegatronStrategyModule


class ForwardForBatch(MegatronStrategyModule):

	def __init__(self, model):
		super(ForwardForBatch, self).__init__(model)
	
	def forward(self, h = None, r = None, t = None):
		if h.dim() == 2 and r.dim() == 1 and t.dim() == 1:
			# check the shape of tensors: h - [B, N] r - [B] t - [B]
			assert t.size(0) == r.size(0) and h.size(0) == r.size(0),  'The shapes of input tensors cannot match with each other!'
			# the shape of score is [B, N]
			score = self.model.forward_for_head_batch(h, r, t)

		elif h.dim() == 1 and r.dim() == 1 and t.dim() == 2:
			# check the shape of tensors: h - [B] r - [B] t - [B, N]
			assert t.size(0) == r.size(0) and h.size(0) == r.size(0),  'The shapes of input tensors cannot match with each other!'
			# the shape of score is [B, N]
			score = self.model.forward_for_tail_batch(h, r, t)
		else:
            # the shape is invalid
			raise Exception('The shapes of input tensors are invalid!')
		
		return score