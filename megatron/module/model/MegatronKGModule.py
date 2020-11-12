#coding:utf-8

import torch

from ..MegatronBaseModule import MegatronBaseModule

class MegatronKGModule(MegatronBaseModule):

	def __init__(self):
		super(MegatronKGModule, self).__init__()

	def forward_for_tail_all(self, h, r):
		raise NotImplementedError

	def forward_for_head_all(self, t, r):
		raise NotImplementedError

	def forward_for_head_batch(self, h, r, t):
		raise NotImplementedError

	def forward_for_tail_batch(self, h, r, t):
		raise NotImplementedError

	def forward_for_triple(self, h, r, t):
		raise NotImplementedError
	
	def predict(self, h = None, r = None, t = None):
		raise NotImplementedError

