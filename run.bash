# export MASTER_ADDR="192.168.1.6"   # 主节点的 IP
# export MASTER_PORT=29501       # 通信端口，保证所有节点一致
# export NNODES=1                # 节点总数
# export NPROC_PER_NODE=1        # 每个节点的 GPU 数量
# export NODE_RANK=0             # 当前节点编号 (主节点=0，从节点=1,...)
# export GLOO_SOCKET_IFNAME=eno2
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eno2  # 或你机器的网卡名
# export NCCL_P2P_DISABLE=0
# export NCCL_DEBUG=INFO
# export TRANSFORMERS_OFFLINE=1

# echo ${MASTER_ADDR}

# torchrun \
#     --nnodes 2 \
#     --nproc_per_node 1 \
#     --node_rank 0 \
#     --master_addr "192.168.1.6" \
#     --master_port 29501 \
#     bin/trainer.py fit -c "configs/clip-graph/inductive-causal/pubmed/base.yaml"; \

# python  bin/trainer.py fit -c "configs/clip-graph/inductive-causal/pubmed/gassobase.yaml"; 
python  bin/trainer.py fit -c "configs/gnn-pretrain/pubmed/gasso-base-lr-1e-2-20epoch.yaml"; 