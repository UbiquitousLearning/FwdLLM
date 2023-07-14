FL_ALG=FedAvg
PARTITION_METHOD="niid_label_clients=30_alpha=1.0"
C_LR=0.1
S_LR=0.1
MU=1
ROUND=5000
# USE_QUANTIZE=$2
# FREEZE_LAYERS=$3

# LOG_FILE="./tmp/onto_${PARTITION_METHOD}_quantize=${USE_QUANTIZE}_freeze=${FREEZE_LAYERS}=.log"
WORKER_NUM=15
CI=0

DATA_DIR=~/fednlp_data/
DATA_NAME=onto  # dataname
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
python -m fedavg_main_st \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key mapping_myMap \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --ci $CI \
  --dataset "${DATA_NAME}" \
  --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
  --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
  --partition_method $PARTITION_METHOD \
  --fl_algorithm $FL_ALG \
  --model_type bert \
  --model_name bert-base-uncased \
  --do_lower_case True \
  --train_batch_size 4 \
  --eval_batch_size 8 \
  --max_seq_length 256 \
  --lr $C_LR \
  --server_lr $S_LR \
  --fedprox_mu $MU \
  --epochs 1 \
  --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
  --use_quantize True \
  # --freeze_layers "e,0,1,2,3,4,5" \
  # --fp16 \
  # --use_adapter "" \
  # > ${LOG_FILE} 2>&1

# sh run_seq_tagging.sh FedAvg "niid_cluster_clients=100_alpha=5.0" 1e-5 0.1 0.5 30

# sh run_seq_tagging.sh FedProx "niid_cluster_clients=100_alpha=5.0" 1e-5 0.1 0.5 30

# sh run_seq_tagging.sh FedOPT "niid_cluster_clients=100_alpha=5.0" 1e-5 0.1 0.5 30