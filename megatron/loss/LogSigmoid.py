#coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.module import MegatronModule

class LogSigmoid(MegatronModule):

	def __init__(self, model, adv_T = None):
		super(LogSigmoid, self).__init__()
		self.model = model
		self.criterion = nn.LogSigmoid()
		if adv_T is None:
			self.adv_T = None
		else:
			self.adv_T = nn.Parameter(torch.Tensor([adv_T]))
			self.adv_T.requires_grad = False

	def get_weight(self, score):
		with torch.no_grad():
			weight =  F.softmax(score * self.adv_T, dim = -1)
			return weight.detach()

	def forward(self, h_idx = None, r_idx = None, t_idx = None):
		score = self.model(h_idx, r_idx, t_idx)
		p_score = score[:,:1]
		n_score = score[:,1:]
		loss = -self.criterion(p_score).mean() - self.criterion(-n_score).mean() if self.adv_T is None else \
			   -self.criterion(p_score).mean() - (self.get_weights(n_score) * self.criterion(-n_score)).sum(-1).mean()
		return 0.5 * loss
			
	def predict(self, h_idx = None, r_idx = None, t_idx = None):
		return self.model(h_idx, r_idx, t_idx)
		# score = self.forward(h_idx, r_idx, t_idx)
		# return score.cpu().data.numpy()
