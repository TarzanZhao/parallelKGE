# coding=utf-8

from megatron.module.model import TransEModel
from megatron.module.strategy import ForwardForBatch
from megatron.module.loss import LogSigmoid
from megatron.training import pretrain

def model_provider():
    model = TransEModel(p_norm = 1, margin = 9.0)
    model = ForwardForBatch(model)
    model = LogSigmoid(model)
    return model

if __name__ == "__main__":
    pretrain(model_provider, args_defaults={})
