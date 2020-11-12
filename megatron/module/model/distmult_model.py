# coding=utf-8

import torch

from megatron import get_args
from megatron import mpu
from .embedding import Embedding
from .MegatronKGModule import MegatronKGModule

class DistMultModel(MegatronKGModule):
    """Bert Language model."""

    def __init__(self):
        super(DistMultModel, self).__init__()
        args = get_args()

        init_method = init_method_normal(args.init_method_std)

        self.embedding = Embedding(
                 hidden_size = args.hidden_size,
                 entity_size = args.entity_size,
                 relation_size = args.relation_size,
                 embedding_dropout_prob = args.embedding_dropout_prob,
                 init_method = init_method,
                 parallel_output = True)

    def _trans_h_to_t(self, h_emb, r_emb):
        tmp_emb = h_emb * r_emb
        # if not self.hidden_batch_norm:
            # tmp_emb = self.bn1(tmp_emb)
        # if not self.hidden_drop is None:
            # tmp_emb = self.hidden_drop(tmp_emb)
        return tmp_emb

    def _trans_t_to_h(self, t_emb, r_emb):
        tmp_emb = t_emb * r_emb
        # if not self.hidden_batch_norm:
        #     tmp_emb = self.bn1(tmp_emb)
        # if not self.hidden_drop is None:
        #     tmp_emb = self.hidden_drop(tmp_emb)
        return tmp_emb

    # def forward_for_tail_all(self, h_idx, r_idx):
    #     h_emb, r_emb, _ = self.embedding(h_idx, r_idx, None)
    #     hr_emb = self._trans_h_to_t(h_emb, r_emb)
    #     score = self.l1_distance_cross(hr_emb, self.ent_embeddings.weight) if p_norm == 1 else \
    #             self.l2_distance_cross(hr_emb, self.ent_embeddings.weight)
    #     if self.parallel_output:
    #         return all_reduce(score)
    #     else:
    #         return score

    # def forward_for_head_all(self, t_idx, r_idx):
    #     _, r_emb, t_emb = self.embedding(None, r_idx, t_idx)
    #     tr_emb = self._trans_t_to_h(t_emb, r_emb)
    #     score = self.l1_distance_cross(tr_emb, self.ent_embeddings.weight) if p_norm == 1 else \
    #             self.l2_distance_cross(tr_emb, self.ent_embeddings.weight)
    #     if self.parallel_output:
    #         return all_reduce(score)
    #     else:
    #         return score

    def forward_for_head_batch(self, h_idx, r_idx, t_idx):
        h_emb, r_emb, t_emb = self.embedding(h_idx, r_idx, t_idx)
        tr_emb = self._trans_t_to_h(t_emb, r_emb)
        score = torch.sum(tr_emb * h_emb, dim = -1)
        return torch.distributed.all_reduce(score)

    def forward_for_tail_batch(self, h_idx, r_idx, t_idx):
        h_emb, r_emb, t_emb = self.embedding(h_idx, r_idx, t_idx)
        hr_emb = self._trans_h_to_t(h_emb, r_emb)
        score = torch.sum(tr_emb * h_emb, dim = -1)
        return torch.distributed.all_reduce(score)


    # def forward_for_triple(self, h_idx, r_idx, t_idx):
    #     h_emb, r_emb, t_emb = self.embedding(h_idx, r_idx, t_idx)
    #     hr_emb = self._trans_h_to_t(h_emb, r_emb)
    #     score = self.l1_distance(hr_emb, t_emb) if p_norm == 1 else \
    #             self.l2_distance(hr_emb, t_emb)
    #     if self.parallel_output:
    #         return all_reduce(score)
    #     else:
    #         return score

