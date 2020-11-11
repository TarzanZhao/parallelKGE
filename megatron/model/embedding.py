# coding=utf-8

import torch

import megatron
from megatron import get_args
from megatron import mpu
from ..module import MegatronModule

class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 hidden_size,
                 entity_size,
                 relation_size,
                 embedding_dropout_prob,
                 init_method,
                 parallel_output = False):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.vocab_size = self.entity_size + self.relation_size

        self.word_embeddings = mpu.ParallelEmbedding(
            self.vocab_size, self.hidden_size, init_method=self.init_method, parallel_output=parallel_output)
        self._word_embeddings_key = 'word_embeddings'

        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def forward(self, head_ids, rel_ids, tail_ids):
        head_ids_shape = (list)(head_ids.size())+[-1]
        rel_ids_shape = (list)(rel_ids.size())+[-1]
        tail_ids_shape = (list)(tail_ids.size())+[-1]

        head_ids = head_ids.flatten()
        tail_ids = tail_ids.flatten()
        rel_ids = rel_ids.flatten() + self.entity_size

        head_ids_elements = head_ids.size(0)
        tail_ids_elements = tail_ids.size(0)
        rel_ids_elements = rel_ids.size(0)

        input_ids = torch.cat([head_ids, tail_ids, rel_ids], 0)
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.embedding_dropout(embeddings)
        head_emb = embeddings[:head_ids_elements,:].view(head_ids_shape)
        tail_emb = embeddings[head_ids_elements:head_ids_elements + tail_ids_elements,:].view(tail_ids_shape)
        rel_emb = embeddings[head_ids_elements + tail_ids_elements:,:].view(rel_ids_shape)
        return head_emb, rel_emb, tail_emb

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""
        state_dict_ = {}
        state_dict_[self._word_embeddings_key] \
            = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] \
                        = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)
