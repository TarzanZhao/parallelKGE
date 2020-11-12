export CUDA_VISIBLE_DEVICES=0,1,2,3

WORLD_SIZE=4
MP_SIZE=2

CHECKPOINT_PATH="./checkpoints/transe_fb15k237"
DATA_PATH="./benchmark/FB15K237/"

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr 127.0.0.1 \
                  --master_port 8889"

KG_ARGS="--hidden-size 200 \
        --entity_size 14541\
        --relation_size 237\
        --embedding_dropout_prob 0 "

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ./pretrain_transe.py \
                $KG_ARGS \
                --data-path $DATA_PATH \
                --model-parallel-size $MP_SIZE \
                --lr 1e-2 \
                --init-method-std 0.02\
                --weight-decay 0\
                --train-epochs 1000 \
                --batch-size 2000 \
                --test-batch-size 100 \
                --negative_sample 64\
                --save $CHECKPOINT_PATH \
                --load $CHECKPOINT_PATH \
                --save-interval 500 \