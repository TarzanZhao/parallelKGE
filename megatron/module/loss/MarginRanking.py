#coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from .MegatronLossModule import MegatronLossModule

class MarginRanking(MegatronLossModule):

	def __init__(self, model, margin, adv_T = None):
		super(MarginRanking, self).__init__(model)
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False
		if adv_T is None:
			self.adv_T = None
		else:
			self.adv_T = nn.Parameter(torch.Tensor([adv_T]))
			self.adv_T.requires_grad = False
	
	def get_weight(self, score):
		with torch.no_grad():
			weight =  F.softmax(-score * self.adv_T, dim = -1)
			return weight.detach()

	def forward(self, h_idx = None, r_idx = None, t_idx = None):
		score = self.model(h_idx, r_idx, t_idx)
		p_score = score[:,:1]
		n_score = score[:,1:]
		loss = (torch.max(p_score - n_score, -self.margin)).mean() if self.adv_T is None else \
			   (self.get_weight(n_score) * torch.max(p_score - n_score, -self.margin)).sum(-1).mean()
		return loss + self.margin

	def predict(self, h_idx = None, r_idx = None, t_idx = None):
		score = self.forward(h_idx, r_idx, t_idx)
		return score.cpu().data.numpy()
