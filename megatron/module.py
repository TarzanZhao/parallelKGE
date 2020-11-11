# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron Module"""

import torch


class MegatronModule(torch.nn.Module):
    """Megatron specific extentions of torch Module."""

    def __init__(self):
        super(MegatronModule, self).__init__()

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(destination, prefix, keep_vars)


#coding:utf-8

class MegatronKGModel(MegatronModule):

	def __init__(self):
		super(MegatronKGModel, self).__init__()

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

	def forward(self, h = None, r = None, t = None):

		if t is None:
			# check the shape of tensors: h - [B] r - [B]
			assert h.dim() == r.dim(), 'The shapes of input tensors cannot match with each other!'
			# the shape of score is [B, N], here N is the number of all entities
			score = self.forward_for_tail_all(h, r)

		elif h is None:
			# check the shape of tensors: t - [B] r - [B]
			assert t.dim() == r.dim(), 'The shapes of input tensors cannot match with each other!'
			# the shape of score is [B, N], here N is the number of all entities
			score = self.forward_for_head_all(t, r)

		elif h.dim() == 2 and r.dim() == 1 and t.dim() == 1:
			# check the shape of tensors: h - [B, N] r - [B] t - [B]
			assert t.size(0) == r.size(0) and h.size(0) == r.size(0),  'The shapes of input tensors cannot match with each other!'
			# the shape of score is [B, N]
			score = self.forward_for_head_batch(h, r, t)

		elif h.dim() == 1 and r.dim() == 1 and t.dim() == 2:
			# check the shape of tensors: h - [B] r - [B] t - [B, N]
			assert t.size(0) == r.size(0) and h.size(0) == r.size(0),  'The shapes of input tensors cannot match with each other!'
			# the shape of score is [B, N]
			score = self.forward_for_tail_batch(h, r, t)

		elif h.dim() == r.dim() and r.dim() == t.dim() and t.dim() <= 2:
			# check the shape of tensors: h - [B, N] r - [B, N] t - [B, N] or h - [B] r - [B] t - [B] 
			assert self._compare_tensor_shape(h, r) and self._compare_tensor_shape(t, r), 'The shapes of input tensors cannot match with each other!'
			# the shape of score is [B, N] or [B]
			score = self.forward_for_triple(h, r, t)

		else:
			# the shape is invalid
			raise Exception('The shapes of input tensors are invalid!')

		return score
	
	def predict(self, h = None, r = None, t = None):
		score = self.forward(h, r, t)
		return score.detach()

