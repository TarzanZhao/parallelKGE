# coding=utf-8

from megatron.model import TransEModel
from megatron.loss import LogSigmoid
from megatron.training import pretrain

def model_provider():
    model = TransEModel(p_norm = 1, margin = 9.0)
    model = LogSigmoid(model)
    return model

if __name__ == "__main__":
    pretrain(model_provider, args_defaults={})
