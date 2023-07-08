import yaml
import os
import sys
import logging
import time


# def set_num_v(k):
#     with open ("./configs/optimization/default.yaml",encoding = "utf-8") as f:
#         doc = yaml.safe_load(f)
#         doc["num_v"] = k
#     with open('./configs/optimization/default.yaml','w') as f:
#         yaml.dump(doc,f)
lr_list = [0.01]
# client_num_list = [5,10, 20, 50, 100]
client_num_list = [1000]
method_list = ["FedFwd"]
# method_list = ["FedAvg", "FedSgd", "FedFwd"]
# num_v_list = [1, 2, 5, 10, 20, 50, 100]

run_id = 0
for lr in lr_list:
    for method in method_list:    
        for client_num in client_num_list:      
            print("method = %s  client_num = %s" % (method, client_num))
            os.system('nohup sh run_text_classification.sh  %s %s %s &' % (client_num, lr, method))
            time.sleep(10)
# os.system('nohup python mnist_backprop.py > ./log_param_vs_k/conv_minist/hidden_state_256_lr3e-5_backprop.log 2>&1 &')