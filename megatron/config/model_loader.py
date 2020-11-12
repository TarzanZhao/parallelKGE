import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import mpu
from megatron.module.model import DistributedDataParallel as LocalDDP

def get_model(model_provider_func):
    """Build the model."""
    args = get_args()

    # Build model on cpu.
    model = model_provider_func()

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Wrap model for distributed training."""
    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        model = torchDDP(model, device_ids=[i], output_device=i,
                         process_group=mpu.get_data_parallel_group())
        return model
    if args.DDP_impl == 'local':
        model = LocalDDP(model)
        return model

    raise NotImplementedError('Unknown DDP implementation specified: {}. '
                              'Exiting.'.format(args.DDP_impl))