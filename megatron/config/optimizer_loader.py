import torch
# from apex.optimizers import FusedAdam as Adam
from torch.optim import Adam,SGD
from megatron import get_args
from megatron.model import get_params_for_weight_decay_optimization
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP

def get_optimizer(model):
    """Set up the optimizer."""
    args = get_args()

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (torchDDP, LocalDDP)):
        model = model.module
    param_groups = get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    # Use Adam.
    # optimizer = SGD(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer