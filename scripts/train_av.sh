batch_size=4
gpu_ids=0,1,2,3
gpu_num=4
n_threads=4
lr=1e-4
n_epochs=3
dt="av_data"
eval_flag=0

experiment_name=audio_visual
base_path='experiments_on_av_data/'$experiment_name

CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.run --nproc_per_node=$gpu_num --master_port=25621 train_av_data.py \
    --multiprocessing_distributed \
    --config cfgs/diffusion.yml \
    --gpu_devices $gpu_ids \
    --gpu 1 \
    --data_type $dt \
    --config_file cfgs/audio_visual.py \
    --batch_size $batch_size \
    --n_threads $n_threads \
    --lr_scheduler MultiStepLR \
    --name $experiment_name \
    --root_path ${base_path} \
    --train \
    --pretrain_path experiments_on_dhf1k/visual/weights/best.pth
    # --test




