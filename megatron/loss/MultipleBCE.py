#coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.module import MegatronModule

class MultipleBCE(MegatronModule):

	def __init__(self, model, smoothing = None, adv_T = None):
		super(MultipleBCE, self).__init__(model)

		if adv_T is None:
			self.adv_T = None
		else:
			self.adv_T = nn.Parameter(torch.Tensor([adv_T]))
			self.adv_T.requires_grad = False

		self.smoothing = smoothing
		self.criterion = torch.nn.BCELoss(size_average = False, reduce = False)

	def get_weight(self, score):
		with torch.no_grad():
			weight = F.softmax(score * self.adv_T, dim = -1)
			return weight.detach()

	def forward(self, h_idx = None, r_idx = None, t_idx = None, target = None):
		if not self.smoothing is None:
			with torch.no_grad():
				target = ((1.0 - self.smoothing) * target) + (1.0 / target.size(-1))
		score = self.model(h_idx, r_idx, t_idx)
		loss = self.criterion(score, target)
		if self.adv_T is None:
			loss = loss.mean()
		else:
			weight = self.get_weight(loss)
			loss = (loss * weight).sum(-1).mean()
		return loss

	def predict(self, h_idx = None, r_idx = None, t_idx = None, target = None):
		score = self.forward(h_idx, r_idx, t_idx, target)
		return score.cpu().data.numpy()
