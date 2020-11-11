# coding=utf-8
import argparse
import os


def parse_args(extra_args_provider=None, defaults={},
               ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)
    # Standard arguments.
    parser = _add_network_size_args(parser)
    parser = _add_training_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_data_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_checkpointing_args(parser)

    # Custom arguments.f
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Distributed args.
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    args.model_parallel_size = min(args.model_parallel_size, args.world_size)
    if args.rank == 0:
        print('using world size: {} and model-parallel size: {} '.format(
            args.world_size, args.model_parallel_size))

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    # Check required arguments.
    required_args = ['hidden_size']
    for req_arg in required_args: 
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    if args.lr is not None:
        assert args.min_lr <= args.lr
    # if args.save is not None:
    #     assert args.save_interval is not None

    _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print('-------------------- arguments --------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (32 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print('---------------- end of arguments ----------------', flush=True)


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')

    group.add_argument('--hidden-size', type=int, default=None,
                       help='Tansformer hidden size.')
    group.add_argument('--entity_size', type=int, default=None,
                       help='entity size.')
    group.add_argument('--relation_size', type=int, default=None,
                       help='relation size.')
    return parser

def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size.')
    group.add_argument('--test-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size.')
    group.add_argument('--negative_sample', type=int, default=64,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size.')

                       
    # group.add_argument('--train-iters', type=int, default=None,
    #                    help='Total number of iterations to train over all '
    #                    'training runs.')
    group.add_argument('--train-epochs', type=int, default=None,
                       help='Total number of epochs to train over all '
                       'training runs.')
    group.add_argument('--train_mode', type=str, default="forward_for_tail_batch",help='train mode.')
                       
    return parser


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')
    group.add_argument('--fp32-allreduce', action='store_true',
                       help='All-reduce in fp32')
    return parser

def _add_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')

    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--init-method-std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')

    group.add_argument('--eval-iters', type=int, default=10000,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')

    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--no-save-optim', action='store_true',
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true',
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true',
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true',
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')

    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--warmup', type=float, default=0.01,
                       help='Percentage of total iterations to warmup on '
                       '(.01 = 1 percent of all training iters).')
    group.add_argument('--override-lr-scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-lr-scheduler', action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')
    return parser

def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed settings')

    group.add_argument('--model-parallel-size', type=int, default=1,
                       help='Size of the model parallel.')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch'],
                       help='which DistributedDataParallel implementation to use.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    return parser

def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    # group.add_argument('--attention-dropout', type=float, default=0.1,
    #                    help='Post attention dropout ptobability.')
    # group.add_argument('--hidden-dropout', type=float, default=0.1,
    #                    help='Dropout probability for hidden state transformer.')
    group.add_argument('--weight-decay', type=float, default=0.00,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--embedding_dropout_prob', type=float, default=None,
                       help='embedding_dropout_prob.')
    # group.add_argument('--clip-grad', type=float, default=1.0,
    #                    help='Gradient clipping based on global L2 norm.')

    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')

    group.add_argument('--data-path', type=str, default=None,
                       help='Path to combined dataset to split.')
    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")
    return parser
