#coding:utf-8

from megatron import get_args
from megatron import mpu
from megatron.utils import make_data_loader
from megatron.data.kg_dataset import build_train_valid_test_datasets

def get_data_loader():
    args = get_args()
    if mpu.get_model_parallel_rank() == 0:
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(data_prefix=args.data_path)
        train_dataloader = make_data_loader(train_ds, args.batch_size, args.num_workers)
        valid_dataloader = make_data_loader(valid_ds, args.test_batch_size, args.num_workers)
        test_dataloader = make_data_loader(test_ds, args.test_batch_size, args.num_workers)
        return train_dataloader, valid_dataloader, test_dataloader, train_ds, valid_ds, test_ds
    else:
        return None, None, None, None, None, None
