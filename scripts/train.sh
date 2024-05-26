batch_size=4
gpu_ids=0,1,2,3
gpu_num=4
n_threads=4
lr=1e-4
n_epochs=3
dt="dhf1k" # dhf1k, ucf holly
eval_flag=0

experiment_name=visual
base_path='experiments_on_dhf1k/'$experiment_name

CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.run --nproc_per_node=$gpu_num --master_port=25623 train_dhf1k.py \
    --multiprocessing_distributed \
    --config cfgs/diffusion.yml \
    --gpu_devices $gpu_ids \
    --gpu 1 \
    --data_type $dt \
    --config_file cfgs/visual.py \
    --batch_size $batch_size \
    --n_threads $n_threads \
    --lr_scheduler MultiStepLR \
    --name $experiment_name \
    --root_path ${base_path} \
    --train
    # --test


if [ $eval_flag -eq 1 ]; then
    if [ $dt = "dhf1k" ]; then
        python compute_metrics.py \
        ${base_path}/${dt}_val_results \
        $dt
    elif [ $dt = "ucf" ]; then
        python compute_metrics.py \
        ${base_path}/${dt}_test_samplings \
        $dt
    elif [ $dt = "holly" ]; then
        python compute_metrics.py \
        ${base_path}/${dt}_val_results \
        $dt
    fi
fi
