DATA_DIR=../fednlp_data
python -m data.advanced_partition.niid_label \
--client_number 1000 \
--data_file ${DATA_DIR}/data_files/20news_data.h5 \
--partition_file ${DATA_DIR}/partition_files/20news_partition.h5 \
--task_type text_classification \
--skew_type natural \
--seed 42