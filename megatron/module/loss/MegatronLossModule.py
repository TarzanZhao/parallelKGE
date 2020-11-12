#coding:utf-8

import torch

from ..MegatronBaseModule import MegatronBaseModule

class MegatronLossModule(MegatronBaseModule):

	def __init__(self):
		super(MegatronLossModule, self).__init__()

	def get_weight(self, score):
		raise NotImplementedError

	def forward(self, h_idx = None, r_idx = None, t_idx = None, target = None):
		raise NotImplementedError

	def predict(self, h_idx = None, r_idx = None, t_idx = None, target = None):
		raise NotImplementedError
