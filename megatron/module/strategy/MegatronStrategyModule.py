#coding:utf-8

import torch

from ..MegatronBaseModule import MegatronBaseModule

class MegatronStrategyModule(MegatronBaseModule):

	def __init__(self, model):
		super(MegatronStrategyModule, self).__init__()
		self.model = model