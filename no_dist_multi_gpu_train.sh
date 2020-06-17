# export NCCL_P2P_DISABLE=1
#export NGPUS=4
GPU_IDS=0,1,2,3
CUDA_VISIBLE_DEVICES=${GPU_IDS} python  tools/train.py --config_file "config/melons_aspp.yaml"