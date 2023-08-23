client_num_per_round=$1
LR=$2
FL_ALG=$3
# PARTITION_METHOD="uniform" 20news
# PARTITION_METHOD="niid_label_clients=1000_alpha=10.0" #agnews
# PARTITION_METHOD="niid_label_clients=100_alpha=100" semeval_2010_task8
# PARTITION_METHOD="niid_label_clients=30_alpha=10.0"
PARTITION_METHOD="uniform_client_1000"
C_LR=0.01
S_LR=0.1
ROUND=3000
WORKER_NUM=1
model_type=distilbert
model_name=distilbert-base-uncased
# model_type=bert
# model_name=bert-base-uncased
# model_type=bert
# model_name=bert-large-uncased
# model_type=albert
# model_name=albert-base-v2
# model_type=roberta-large
# model_name=roberta-large
# model_type=deberta
# model_name=microsoft/deberta-xlarge
train_batch_size=8
DATA_NAME=agnews
fold_name=${model_type}_${DATA_NAME}
# frequency_of_the_test=1

if [ $DATA_NAME = "agnews" ];then
  max_seq_length=64
  frequency_of_the_test=1
elif [ $DATA_NAME = "20news" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "yelp-full" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "mnli" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "yelp-p" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "sst_2" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "semeval_2010_task8" ];then
  max_seq_length=128
  frequency_of_the_test=1
elif [ $DATA_NAME = "yahoo" ];then
  max_seq_length=256
  frequency_of_the_test=5
  PARTITION_METHOD="uniform_client_10000"
else
  max_seq_length=256
  frequency_of_the_test=1
fi


LOG_FILE="fedavg_transformer_tc.log"
# WORKER_NUM=10
CI=0

DATA_DIR=/data/wyz/fednlp_data/
# 20news agnews semeval_2010_task8

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file
if [ $FL_ALG = "FedAvg" ];then
  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  python -m fedavg_main_tc \
    --gpu_mapping_file "gpu_mapping.yaml" \
    --gpu_mapping_key mapping_myMap \
    --client_num_per_round $client_num_per_round \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method $PARTITION_METHOD \
    --fl_algorithm $FL_ALG \
    --model_type $model_type\
    --model_name $model_name \
    --do_lower_case True \
    --train_batch_size $train_batch_size \
    --frequency_of_the_test $frequency_of_the_test \
    --eval_batch_size 8 \
    --max_seq_length $max_seq_length \
    --lr $C_LR \
    --server_lr $S_LR \
    --epochs 1 \
    --use_adapter True \
    --learning_rate $LR \
    >/data/wyz/FedNLP/all_figures/fedavg_result/${fold_name}/fedavg_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}_no_adapter.log 2>&1
    # > ./log/${fold_name}/fedavg_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}.log 2>&1
elif [ $FL_ALG = FedSgd ];then
  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  python -m fedavg_main_tc \
    --gpu_mapping_file "gpu_mapping.yaml" \
    --gpu_mapping_key mapping_myMap \
    --client_num_per_round $client_num_per_round \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method $PARTITION_METHOD \
    --fl_algorithm $FL_ALG \
    --model_type $model_type\
    --model_name $model_name \
    --frequency_of_the_test $frequency_of_the_test \
    --do_lower_case True \
    --train_batch_size $train_batch_size \
    --eval_batch_size 8 \
    --max_seq_length $max_seq_length \
    --lr $C_LR \
    --server_lr $S_LR \
    --epochs 1 \
    --use_adapter True \
    --learning_rate $LR \
    > /data/wyz/FedNLP/all_figures/distilbert_agnews_diff_client_num/fedsgd_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}.log 2>&1
    # > ./log/end2end/${fold_name}/fedsgd_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}_full.log 2>&1
else
  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  python -m fedavg_main_tc \
    --gpu_mapping_file "gpu_mapping.yaml" \
    --gpu_mapping_key mapping_myMap \
    --client_num_per_round $client_num_per_round \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method $PARTITION_METHOD \
    --fl_algorithm $FL_ALG \
    --model_type $model_type\
    --model_name $model_name \
    --frequency_of_the_test $frequency_of_the_test \
    --do_lower_case True \
    --train_batch_size $train_batch_size \
    --eval_batch_size 8 \
    --max_seq_length $max_seq_length \
    --lr $C_LR \
    --server_lr $S_LR \
    --worker_num $WORKER_NUM \
    --epochs 1 \
    --use_adapter True \
    --forward_mode \
    --learning_rate $LR \
    > ./log/end2end/${fold_name}/fedFwd_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}_numerical_var_threthod_0.15_adam.log 2>&1
# fi
# mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
# python -m fedavg_main_tc \
#   --gpu_mapping_file "gpu_mapping.yaml" \
#   --gpu_mapping_key mapping_myMap \
#   --client_num_per_round $client_num_per_round \
#   --comm_round $ROUND \
#   --ci $CI \
#   --dataset "${DATA_NAME}" \
#   --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
#   --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
#   --partition_method $PARTITION_METHOD \
#   --fl_algorithm $FL_ALG \
#   --model_type distilbert \
#   --model_name distilbert-base-uncased \
#   --do_lower_case True \
#   --train_batch_size 8 \
#   --eval_batch_size 8 \
#   --max_seq_length 64 \
#   --lr $C_LR \
#   --server_lr $S_LR \
#   --epochs 1 \
#   --use_adapter True \
#   --learning_rate $LR \
#   > ./log/test2.log 2>$1
  # > ./log/fedavg_lr1e-2_client_num_5.log 2>$1
  # > ./log/${DATA_NAME}/forward_lr${LR}_client_num_${client_num_per_round}.log 2>&1
  # > ./log/fedsgd_lr1e-3_client_num_50.log 2>$1
  # --freeze_layers "e,0,1,2,3,4,5,6,7,8,9" \
  # --use_quantize False \
  # --use_adapter "" \
  # --forward_mode \


# sh run_text_classification.sh FedAvg "niid_label_clients=100_alpha=5.0" 5e-5 0.1 50
# sh run_text_classification.sh FedAvg "niid_label_clients=100_alpha=10.0" 5e-5 0.1 50
# sh run_text_classification.sh FedAvg "niid_label_clients=100_alpha=1.0" 5e-5 0.1 50
# sh run_text_classification.sh FedAvg "uniform" 5e-5 0.1 50
# sh run_text_classification.sh FedAvg "niid_quantity_clients=100_beta=5.0" 5e-5 0.1 50

# sh run_text_classification.sh FedProx "niid_label_clients=100_alpha=5.0" 5e-5 0.1 50
# sh run_text_classification.sh FedProx "niid_label_clients=100_alpha=10.0" 5e-5 0.1 50
# sh run_text_classification.sh FedProx "niid_label_clients=100_alpha=1.0" 5e-5 0.1 50
# sh run_text_classification.sh FedProx "uniform" 5e-5 0.1 50
# sh run_text_classification.sh FedProx "niid_quantity_clients=100_beta=5.0" 5e-5 0.1 50

# sh run_text_classification.sh FedOPT "niid_label_clients=100_alpha=5.0" 5e-5 0.1 50
# sh run_text_classification.sh FedOPT "niid_label_clients=100_alpha=10.0" 5e-5 0.1 50
# sh run_text_classification.sh FedOPT "niid_label_clients=100_alpha=1.0" 5e-5 0.1 50
# sh run_text_classification.sh FedOPT "uniform" 5e-5 0.1 300
# sh run_text_classification.sh FedOPT "niid_quantity_clients=100_beta=5.0" 5e-5 0.1 50
fi