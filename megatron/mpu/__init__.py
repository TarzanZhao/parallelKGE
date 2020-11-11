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

# 各个进程获得rank和group信息
from .initialize import destroy_model_parallel
from .initialize import get_data_parallel_group
from .initialize import get_data_parallel_rank
from .initialize import get_data_parallel_world_size
from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_src_rank
from .initialize import get_model_parallel_world_size
from .initialize import initialize_model_parallel
from .initialize import model_parallel_is_initialized
# 数据广播
from .data import broadcast_data
# 线性层、向量层的并行
from .layers import ColumnParallelLinear
from .layers import ParallelEmbedding
from .layers import RowParallelLinear
from .layers import VocabParallelEmbedding
# 基本的模型并行层
from .mappings import copy_to_model_parallel_region
from .mappings import gather_from_model_parallel_region
from .mappings import reduce_from_model_parallel_region
from .mappings import scatter_to_model_parallel_region
# 词典在不同卡上的softmax
from .cross_entropy import vocab_parallel_cross_entropy
# 随机算法
from .random import checkpoint
from .random import get_cuda_rng_tracker
from .random import model_parallel_cuda_manual_seed
# 在model并行上的各个卡上进行数据切分
from .utils import divide
from .utils import split_tensor_along_last_dim
