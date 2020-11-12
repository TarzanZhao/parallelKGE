#coding:utf-8

import torch

from .MegatronStrategyModule import MegatronStrategyModule

class ForwardForAll(MegatronStrategyModule):
    
	def __init__(self, model):
		super(ForwardForAll, self).__init__(model)

	def forward(self, h = None, r = None, t = None):
		if t is None :
			# check the shape of tensors: h - [B] r - [B]
			assert h.dim() == r.dim(), 'The shapes of input tensors cannot match with each other!'
			# the shape of score is [B, N], here N is the number of all entities
			score = self.model.forward_for_tail_all(h, r)
		elif h is None:
			# check the shape of tensors: t - [B] r - [B]
			assert t.dim() == r.dim(), 'The shapes of input tensors cannot match with each other!'
			# the shape of score is [B, N], here N is the number of all entities
			score = self.model.forward_for_head_all(t, r)
		else:
			# the shape is invalid
			raise Exception('The shapes of input tensors are invalid!')

		return score