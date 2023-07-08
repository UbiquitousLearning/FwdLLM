import os
from time import sleep
import logging

lr_list = [1e-2]

# freeze_layers = ["e", "e,0", "e,0,1", "e,0,1,2", "e,0,1,2,3", "e,0,1,2,3,4", "e,0,1,2,3,4,5"]
freeze_layers = ["e", "e,0,1,2"]
# v_num_list = [1,5,20,100,500]
v_num_list = [1]
for lr in lr_list:
    for layer in freeze_layers:
        for v_num in v_num_list:
            print(f"{lr}  {layer} {v_num}")
            os.system('nohup sh run_text_classification_sweep.sh %s %s %s &' % (lr, layer, v_num))
            sleep(3)