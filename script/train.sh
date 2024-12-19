CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 9573 train.py --index Caltech_test --local_rank -1 --model ResNet --basic_model resnet50_relu  --device 1 --dataset_name Caltech101 --margin 0.01 --concept_cha 32 32 32 32 --concept_per_layer 8 16 32 64 --optimizer adam