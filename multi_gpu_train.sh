# export NCCL_P2P_DISABLE=1
#export NGPUS=4
GPU_IDS=0,1,2,3
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch  --nproc_per_node=4  tools/train.py --config_file "config/melons_aspp.yaml"  2>&1|tee train_info.log