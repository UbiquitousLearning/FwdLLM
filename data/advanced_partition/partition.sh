DATA_DIR=/data/wyz/fednlp_data/

python -m data.advanced_partition.niid_label \
--client_number 100 \
--data_file ${DATA_DIR}/data_files/yahoo_data.h5 \
--partition_file ${DATA_DIR}/partition_files/yahoo_partition.h5 \
--task_type text_classification \
--skew_type label \
--seed 42 \
--kmeans_num 0  \
--alpha 0.5