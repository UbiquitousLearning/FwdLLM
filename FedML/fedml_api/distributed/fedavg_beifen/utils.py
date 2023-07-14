from dataclasses import dataclass
from distutils.log import error
from itertools import accumulate
from operator import mod
import os
from re import A
from unittest.mock import mock_open
from sklearn.preprocessing import scale
import logging

import torch
import numpy as np


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def post_complete_message_to_sweep_process(args):
    pipe_path = "./tmp/fedml"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, 'w') as pipe:
        pipe.write("training is finished! \n%s\n" % (str(args)))

# def quantize_params(model_params_list,accumulated_error=None):
    
#     for k in model_params_list.keys():
#         if "position_ids" not in k and "embeddings" not in k and "0" not in k and "1" not in k and "2" not in k and "3" not in k and "4" not in k and "5" not in k and "6" not in k and "7" not in k and "8" not in k and "9" not in k:
#         # if "position_ids" not in k:
#         #     model_params_list[k] = quantize_tensor(model_params_list[k])
#         # else:
#             model_params_list[k],accumulated_error = quantize_tensor(model_params_list[k])
#         # if "weight" in k and "LayerNorm" not in k and "layer_norm" not in k and "embedding" not in k:
#         #     model_params_list[k] = quantize_tensor_with_bucket(model_params_list[k])
#         # model_params_list[k] = quantize_tensor(model_params_list[k])
#     return model_params_list,accumulated_error
def quantize_params(model_params_list,accumulated_error=None):
    if accumulated_error is None:
        accumulated_error = {}
        for k in model_params_list.keys():
            if "position_ids" not in k and "embeddings" not in k and "0" not in k and "1" not in k and "2" not in k and "3" not in k and "4" not in k and "5" not in k and "6" not in k and "7" not in k and "8" not in k and "9" not in k:
            # if "position_ids" not in k and "embeddings" not in k and "0" not in k and "1" not in k and "2" not in k and "3" not in k and "4" not in k and "5" not in k:
            # if "position_ids" not in k:    
                model_params_list[k],error = quantize_tensor(model_params_list[k])
                accumulated_error[k] = error
    else:   
        for k in model_params_list.keys():
            if "position_ids" not in k and "embeddings" not in k and "0" not in k and "1" not in k and "2" not in k and "3" not in k and "4" not in k and "5" not in k and "6" not in k and "7" not in k and "8" not in k and "9" not in k:
            # if "position_ids" not in k and "embeddings" not in k and "0" not in k and "1" not in k and "2" not in k and "3" not in k and "4" not in k and "5" not in k:
            # if "position_ids" not in k:    
                model_params_list[k],error = quantize_tensor(model_params_list[k]+accumulated_error[k])
                accumulated_error[k] = error
    return model_params_list,accumulated_error

def dequantize_params(model_params_list):
    for k in list(model_params_list.keys()):
        if "position_ids" not in k and "embeddings" not in k and "0" not in k and "1" not in k and "2" not in k and "3" not in k and "4" not in k and "5" not in k and "6" not in k and "7" not in k and "8" not in k and "9" not in k:
        # if "position_ids" not in k and "embeddings" not in k and "0" not in k and "1" not in k and "2" not in k and "3" not in k and "4" not in k and "5" not in k:
        # if "position_ids" not in k:
    #   if "weight" in k and "LayerNorm" not in k and "layer_norm" not in k and "embedding" not in k:
    #         model_params_list[k] = dequantize_tensor_with_bucket(model_params_list[k])
            model_params_list[k] = dequantize(model_params_list[k])
    return model_params_list

class quantized_tensor:
    def __init__(self,data,scale,zero_point) -> None:
        self.data = data
        self.scale = scale
        self.zero_point = zero_point


def quantize_tensor(data):
    scale = (data.max() - data.min())/14
    zero_point = (data.max() + data.min())/2
    quantized_data = torch.round(((data-torch.full_like(data,zero_point))/torch.full_like(data,scale)))
    accumulated_error = data-(quantized_data*scale+zero_point)
    quantized_data = quantized_data.type(torch.int8)
    return quantized_tensor(quantized_data,scale,zero_point),accumulated_error

# def quantize_tensor16(data):
#     scale = (data.max() - data.min())/65534
#     zero_point = (data.max() + data.min())/2
#     data = torch.round(((data-torch.full_like(data,zero_point))/torch.full_like(data,scale))).type(torch.int16)
#     return quantized_tensor(data,scale,zero_point)

def dequantize(qt):
    # return qt.data.type(torch.float32)*torch.full_like(qt.data,qt.scale,dtype=torch.float32)+torch.full_like(qt.data,qt.zero_point,dtype=torch.float32)
    return qt.data.type(torch.float32)*qt.scale+qt.zero_point

class quantized_tensor_with_bucket:
    def __init__(self,data,size) -> None:
        self.data = data
        self.size = size

def quantize_tensor_with_bucket(data):
    size = data.size()
    data = [quantize_tensor(d) for d in data.view(-1,512)]
    return quantized_tensor_with_bucket(data,size)

def dequantize_tensor_with_bucket(data):
    size = data.size
    data = data.data
    return torch.stack([dequantize(d) for d in data]).view(size)