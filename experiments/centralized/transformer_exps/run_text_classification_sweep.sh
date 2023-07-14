GPU_NUM=0,1,2,3,4,5,6,7
lr=$1
LAYERS=$2
v_num=$3
python=/home/wuyaozong/.conda/envs/fwdgrad_py38/bin/python
DATA_NAME=agnews
CUDA_VISIBLE_DEVICES=$GPU_NUM ${python} -m main_tc \
    --dataset ${DATA_NAME} \
    --data_file /data/wyz/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file /data/wyz/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method uniform_client_1000 \
    --model_type albert \
    --model_name albert-base-v2 \
    --do_lower_case True \
    --train_batch_size 80 \
    --eval_batch_size 8 \
    --max_seq_length 64 \
    --learning_rate ${lr} \
    --epochs 10 \
    --evaluate_during_training_steps 18 \
    --output_dir /tmp/${DATA_NAME}_fed/ \
    --use_adapter True \
    --n_gpu 1 \
    --v_num ${v_num} \
    > ./log/warmup_new_gpu_${DATA_NAME}_distilbert_forward_lr=${lr}_layer=${LAYERS}_v_num=${v_num}.log 2>&1
    # --forward_mode \
        # --freeze_layers ${LAYERS} \
    # --freeze_layers "e,0,1,2,3,4,5,6,7,8,9,10,11" \