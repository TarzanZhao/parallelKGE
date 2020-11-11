import torch
import tqdm
from megatron import get_args
from megatron import print_rank_0
from megatron import mpu
from megatron.mpu.initialize import get_model_parallel_rank
from megatron.utils import reduce_losses
from megatron.config import initialize_megatron, get_model, get_optimizer, get_learning_rate_scheduler, get_data_loader
from megatron.checkpoint import load_checkpoint
from megatron.checkpoint import save_checkpoint
from tqdm import tqdm

def pretrain(model_provider_func, extra_args_provider=None, args_defaults={}):

    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    args = get_args()
    model = get_model(model_provider_func)
    optimizer = get_optimizer(model)
    lr_scheduler = get_learning_rate_scheduler(optimizer)
    if args.train_epochs > 0:
        train(model, optimizer, lr_scheduler)


def backward_step(optimizer, model, loss):
    """Backward step."""
    args = get_args()
    optimizer.zero_grad()
    loss.backward()
    # for name, param in model.named_parameters():
    #     if param.requires_grad and param.grad is not None:


    # if args.DDP_impl == 'local':
    #     model.allreduce_params(reduce_after=False,
    #                            fp32_allreduce=args.fp32_allreduce)

def get_test_batch_rank0(index = None, _data = None):
    wargs = get_args()
    keys = ['flag', 'head', 'rel', 'tail']
    datatype = torch.int64
    data = {}

    if _data is not None:
        data['flag'] = torch.cuda.LongTensor([1])
        data['head'] = _data[:,0].cuda()
        data['rel'] = _data[:,1].cuda()
        data['tail'] = _data[:,2].cuda()
        neg = torch.arange(start = 0, end = args.entity_size, device = torch.device('cuda'))
        neg = neg.repeat(data['head'].size(0), 1)
        if index % 2 == 0:
            data['head'] = torch.cat([data['head'].unsqueeze(1), neg], 1)
        else:
            data['tail'] = torch.cat([data['tail'].unsqueeze(1), neg], 1)
    else:
        data['flag'] = torch.cuda.LongTensor([0])
        data['head'] = torch.cuda.LongTensor([0])
        data['rel'] = torch.cuda.LongTensor([0])
        data['tail'] = torch.cuda.LongTensor([0])
    
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    flag = data_b['flag'].long()
    head = data_b['head'].long()
    tail = data_b['tail'].long()
    rel = data_b['rel'].long()

    return flag, head, rel, tail

def get_batch_rank0(index = None, _data = None):
    args = get_args()
    keys = ['flag', 'head', 'rel', 'tail']
    datatype = torch.int64
    data = {}

    if _data is not None:
        data['flag'] = torch.cuda.LongTensor([1])
        data['head'] = _data[:,0].cuda()
        data['rel'] = _data[:,1].cuda()
        data['tail'] = _data[:,2].cuda()
        neg = torch.randint(low = 0, high = args.entity_size, 
        size = (args.batch_size, args.negative_sample), device = torch.device('cuda'))
        if index % 2 == 0:
            data['head'] = torch.cat([data['head'].unsqueeze(1), neg], 1)
        else:
            data['tail'] = torch.cat([data['tail'].unsqueeze(1), neg], 1)
    else:
        data['flag'] = torch.cuda.LongTensor([0])
        data['head'] = torch.cuda.LongTensor([0])
        data['rel'] = torch.cuda.LongTensor([0])
        data['tail'] = torch.cuda.LongTensor([0])
    
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    flag = data_b['flag'].long()
    head = data_b['head'].long()
    tail = data_b['tail'].long()
    rel = data_b['rel'].long()

    return flag, head, rel, tail

def get_batch():
    args = get_args()
    keys = ['flag', 'head', 'rel', 'tail']
    datatype = torch.int64
    data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    flag = data_b['flag'].long()
    head = data_b['head'].long()
    tail = data_b['tail'].long()
    rel = data_b['rel'].long()

    return flag, head, rel, tail


def forward_step(head, rel, tail, model):
    loss = model(head, rel, tail)
    reduced_losses = reduce_losses([loss]) #???
    return loss, {'loss': reduced_losses[0]}


def train_step(head, rel, tail, model,
                optimizer, lr_scheduler):
    loss, loss_reduced = forward_step(head, rel, tail, model)
    backward_step(optimizer, model, loss)
    optimizer.step()
    return loss_reduced


def test(model, test_data_iterator, test_dataseet):
    args = get_args()
    model.eval()
    if get_model_parallel_rank() == 0:
        hit10 = 0.0
        total = 0.0
        for index, _data in enumerate(tqdm(iter(test_data_iterator))):
            flag, head, rel, tail = get_test_batch_rank0(0, _data)
            res = model.predict(head, rel, tail)
            for index, (r, t) in enumerate(zip(rel, tail)):
                target_score = res[index][0]
                all_score = res[index][1:]
                filter_flag = test_dataseet.h_of_tr(t.cpu().numpy(), r.cpu().numpy())
                for flag in filter_flag:
                    all_score[flag] = -21474836
                final = (all_score >= target_score).sum()
                if final < 10:
                    hit10+=1
                total+=1.0
            flag, head, rel, tail = get_test_batch_rank0(1, _data)
            res = model.predict(head,rel,tail)
            for index, (r, h) in enumerate(zip(rel, head)):
                target_score = res[index][0]
                all_score = res[index][1:]
                filter_flag = test_dataseet.t_of_hr(h.cpu().numpy(), r.cpu().numpy())
                for flag in filter_flag:
                    all_score[flag] = -21474836
                final = (all_score > target_score).sum()
                if final < 10:
                    hit10+=1
                total+=1.0
            print (hit10/total)
        get_test_batch_rank0()
    else:
        while True:
            flag, head, rel, tail = get_batch()
            if flag[0] < 1:
                break
            res = model.predict(head,rel,tail)
            flag, head, rel, tail = get_batch()
            if flag[0] < 1:
                break
            res = model.predict(head,rel,tail)
    model.train()

def train(model, optimizer, lr_scheduler):
    """Train the model function."""
    args = get_args()
    model.train()
    train_data_iterator, valid_data_iterator, test_data_iterator, _, _, test_dataseet = \
        get_data_loader()
    
    last_epoch = 0
    last_epoch = load_checkpoint(model, optimizer, lr_scheduler)

    if get_model_parallel_rank() == 0:
        for epoch in tqdm(range(last_epoch, args.train_epochs)):
            res = 0.0
            for index, _data in enumerate(iter(train_data_iterator)):
                flag, head, rel, tail = get_batch_rank0(index, _data)
                loss_dict = train_step(head, rel, tail, model,
                                optimizer, lr_scheduler)
                res = res + loss_dict['loss']
            get_batch_rank0()
            print_rank_0(res)
            # if (epoch + 1) % args.save_interval == 0:
            #     save_checkpoint(epoch+1, model, optimizer, lr_scheduler)
        test(model, test_data_iterator, test_dataseet)
    else:
        for epoch in range(last_epoch, args.train_epochs):
            while True:
                flag, head, rel, tail = get_batch()
                if flag[0] < 1:
                    break
                loss_dict = train_step(head, rel, tail, model,
                                optimizer, lr_scheduler)
            # if (epoch + 1) % args.save_interval == 0:
            #     save_checkpoint(epoch+1, model, optimizer, lr_scheduler)
        test(model, test_data_iterator, test_dataseet)