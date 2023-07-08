# coding: utf-8

from __future__ import absolute_import, division, print_function

import copy
from hashlib import shake_128
import logging
import math
import os
import time
from typing import Callable, Tuple
import collections

import numpy as np
import sklearn
import torch
# import wandb
from torch import nn
from training.utils.text_classification_utils import *
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from functools import partial
import functorch as fc
from torch.cuda.amp import autocast


class ForwardTextClassificationTrainer:
    def __init__(self, args, device, model, train_dl=None, test_dl=None):
        self.args = args
        self.device = device

        # set data
        self.num_labels = args.num_labels
        self.set_data(train_dl, test_dl)

        # model
        self.model = model
        if self.args.model_type == "distilbert":
            self.model.add_module('pre_classifier',nn.Sequential())
        self.model.to(self.device)
        # for p in self.model.modules():
        #     if isinstance(p,torch.nn.modules.normalization.LayerNorm):
        #         p.elementwise_affine = False

        # training results
        self.results = {}
        self.best_accuracy = 0.0

        # freeze
        self.freeze_layers = args.freeze_layers.split(",") if args.freeze_layers else []

        self.grad = None

        self.mode_counter = 0

        # self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(model)

    def set_data(self, train_dl=None, test_dl=None):
        # Used for fedtrainer
        self.train_dl = train_dl
        self.test_dl = test_dl
    
    def generate_v_with_cos(self,min_cos,max_cos,target_grad,v_num):
        batch_size = 100

        # 计算总批次数
        num_batches = math.ceil(v_num / batch_size)

        # 创建一个空的结果张量
        result = torch.empty((v_num,*target_grad.shape))

        # 逐批次计算余弦相似度
        for i in range(num_batches):
            # 获取当前批次的起始索引和结束索引
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, v_num)

            selected_cos = ((max_cos - min_cos) * torch.rand(v_num, 1) + min_cos).to(self.device)

            # 随机向量
            r = torch.stack([torch.randn_like(target_grad) for _ in range(v_num)],dim=0)

            # 根据随机向量算出的垂直grad的向量
            grad_perp = r - (r@(target_grad.unsqueeze(-1)))*target_grad
            # Make it a unit vector:
            grad_perp = grad_perp / (grad_perp.norm(dim=-1)[:,None])

            # 算出v
            candidate_v = selected_cos*target_grad + torch.sqrt(torch.tensor(1,device=self.device) - selected_cos**2)*grad_perp
            candidate_v = candidate_v*torch.tensor((target_grad.shape[0]**0.5),device=self.device)

            # 将结果保存到结果张量的对应位置
            result[start_idx:end_idx] = candidate_v.to("cpu")

            return result

    def train_model(self, device=None):
        if not device:
            device = self.device

        logging.info("train_model self.device: " + str(device))
        self.model.to(device)

        # build optimizer and scheduler
        iteration_in_total = len(
            self.train_dl) // self.args.gradient_accumulation_steps * self.args.epochs
        optimizer, scheduler = self.build_optimizer(self.model, iteration_in_total)
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)

        v_num = len(self.train_dl)
        v_buffer = {}
        index = 0
        if self.mode_counter:
            with torch.no_grad():   
                for k,value in self.model.named_parameters():
                    torch.cuda.empty_cache()
                    # logging.info(index)
                    if self.grad != None:
                        if value.requires_grad:
                            target_grad = self.grad[index]
                            target_grad = torch.flatten(target_grad)
                            target_grad = target_grad/target_grad.norm()

                            candidate_v = self.generate_v_with_cos(0.1,0.5,target_grad,v_num)
                        
                            v_shape = list(value.shape)
                            v_shape.insert(0,v_num)
                            v_buffer[index] = candidate_v.reshape(v_shape).to("cpu")
                        # else:
                        #     v_buffer[index] = torch.stack([torch.zeros_like(value) for _ in range(v_num)],dim=0).to("cpu")
                    # else:
                    #     # logging.info("here")
                    #     v_buffer[index] = torch.stack([torch.randn_like(value) if value.requires_grad else torch.zeros_like(value) for _ in range(v_num)],dim=0).to("cpu")
                    index += 1
        
        

        # training result
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        if self.args.fl_algorithm == "FedProx":
            global_model = copy.deepcopy(self.model)

        self.grad = [torch.zeros_like(p) for p in self.params]

        with torch.no_grad():
            for epoch in range(0, self.args.epochs):
                for batch_idx, batch in enumerate(self.train_dl):

                    batch = tuple(t for t in batch)
                    x = batch[1].to(device)
                    labels = batch[4].to(device)

                    # if self.mode_counter:
                    if self.mode_counter == 0 or v_buffer == {}:
                        v_params = tuple([torch.randn_like(p) if p.requires_grad == True else torch.zeros_like(p) for p in self.params])
                    else:
                        v_params = tuple([v_buffer[i][batch_idx].to(self.device) if p.requires_grad else torch.zeros_like(p) for i,p in enumerate(self.params)])
             
                    f = partial(
                        functional_get_loss,
                        model=self.fmodel,
                        buffers = self.buffers,
                        num_classes = self.num_labels,
                        x=x,
                        t=labels,
                    )

                    # Forward AD
                    # loss, jvp = fc.jvp(f, (self.params,), (v_params,))

                    h = 0.01

                    with autocast():
                        loss = f(tuple([self.params[i]-h*v_params[i] for i in range(len(self.params))]))
                        terbulence_loss = f(tuple([self.params[i]+h*v_params[i] for i in range(len(self.params))]))
                    jvp = (terbulence_loss - loss)/(2*h)
                    
                    for j, fg in enumerate(self.grad):
                        fg.add_(jvp*v_params[j])


                    current_loss = loss.item()
                    logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
                                                                            len(self.train_dl), current_loss))
                    global_step += 1
                    if self.args.evaluate_during_training and (self.args.evaluate_during_training_steps > 0
                                                                and global_step!=0  and global_step % self.args.evaluate_during_training_steps == 0):
                        results, _, _ = self.eval_model(epoch, global_step)

                    if self.args.is_debug_mode == 1 and global_step > 3:
                        break
        
        self.mode_counter += 1
        if self.mode_counter == 2:
            self.mode_counter = 0

        return global_step, tr_loss / global_step

    def eval_model(self, epoch=0, global_step=0, device=None):
        if not device:
            device = self.device

        results = {}

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(self.test_dl)
        test_sample_len = len(self.test_dl.dataset)
        preds = np.empty((test_sample_len, self.num_labels))

        out_label_ids = np.empty(test_sample_len)
        self.model.to(device)
        self.model.eval()
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)
        logging.info("len(test_dl) = %d, n_batches = %d" % (len(self.test_dl), n_batches))
        for i, batch in enumerate(self.test_dl):
            with torch.no_grad():
                batch = tuple(t.to(device) for t in batch)
                # sample_index_list = batch[0].cpu().numpy()
                # if i == len(self.test_dl) - 1:
                #     logging.info(batch)
                x = batch[1]
                labels = batch[4]

                # output = self.fmodel(self.params,self.buffers,x)
                output = self.model(x)
                logits = output[0]

                
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()
                # logging.info("test. batch index = %d, loss = %s" % (i, str(eval_loss)))

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else test_sample_len
            # logging.info("batch index = %d, start_index = %d, end_index = %d" % (i, start_index, end_index))
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps

        model_outputs = preds
        preds = np.argmax(preds, axis=1)
        # logging.info("preds = " + str(preds))
        # logging.info("out_label_ids = " + str(out_label_ids))
        result, wrong = self.compute_metrics(preds, out_label_ids, self.test_dl.examples)
        result["eval_loss"] = eval_loss
        results.update(result)

        self.results.update(result)
        logging.info(self.results)

        return result, model_outputs, wrong
    
    def quick_test(self, epoch=0, global_step=0, device=None,test_num=1000):
        
        if not device:
            device = self.device

        results = {}

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(self.test_dl)
        test_sample_len = test_num * self.args.eval_batch_size
        preds = np.empty((test_sample_len, self.num_labels))

        out_label_ids = np.empty(test_sample_len)
        self.model.to(device)
        self.model.eval()
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)
        logging.info("len(test_dl) = %d, n_batches = %d" % (len(self.test_dl), n_batches))
        for i, batch in enumerate(self.test_dl):
            if i == test_num:
                break
            with torch.no_grad():
                batch = tuple(t.to(device) for t in batch)
                x = batch[1]
                labels = batch[4]

                # output = self.fmodel(self.params,self.buffers,x)
                output = self.model(x)
                logits = output[0]

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else test_sample_len
            # logging.info("batch index = %d, start_index = %d, end_index = %d" % (i, start_index, end_index))
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

        preds = np.argmax(preds, axis=1)
        # logging.info("preds = " + str(preds))
        # logging.info("out_label_ids = " + str(out_label_ids))
        result, wrong = self.compute_metrics(preds, out_label_ids, self.test_dl.examples)
        results.update(result)

        self.results.update(result)
        logging.info(f"quick test accuracy is : {results['acc']}")

        return results['acc']
    
    def quick_test_loss(self, epoch=0, global_step=0, device=None,test_num=1000):
        
        if not device:
            device = self.device

        eval_loss = 0.0
        nb_eval_steps = 0
    
        self.model.to(device)
        self.model.eval()
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)
        for i, batch in enumerate(self.test_dl):
            if i == test_num:
                break
            with torch.no_grad():
                batch = tuple(t.to(device) for t in batch)
                # sample_index_list = batch[0].cpu().numpy()
                # if i == len(self.test_dl) - 1:
                #     logging.info(batch)
                x = batch[1]
                labels = batch[4]

                # output = self.fmodel(self.params,self.buffers,x)
                output = self.model(x)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        logging.info(f"quick test loss is : {eval_loss}")

        return eval_loss

    def compute_metrics(self, preds, labels, eval_examples=None):
        assert len(preds) == len(labels)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )

    def build_optimizer(self, model, iteration_in_total):
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        self.args.warmup_steps = warmup_steps if self.args.warmup_steps == 0 else self.args.warmup_steps
        logging.info("warmup steps = %d" % self.args.warmup_steps)
        # freeze exps only apply for distilbert
        # if self.args.model_type == "distilbert" or self.args.model_type == "bert":
        self.freeze_model_parameters(model)
        if self.args.fl_algorithm == "FedOPT" or self.args.fl_algorithm == "":
            optimizer = AdamW(model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        else:
            optimizer = SGD(model.parameters(), lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=iteration_in_total
        )
        logging.info(get_parameter_number(model))
        return optimizer, scheduler
    
    def freeze_model_parameters(self, model):
        modules = list()
        logging.info("freeze layers: %s" % str(self.freeze_layers))
        if self.args.model_type == "distilbert":
            for layer_idx in self.freeze_layers:
                if layer_idx == "e":
                    modules.append(model.distilbert.embeddings)
                else:
                    modules.append(model.distilbert.transformer.layer[int(layer_idx)])
            # modules.append(model.pre_classifier)
            # for module in model.modules():
            #     print(module)
        elif self.args.model_type == "bert":
            for layer_idx in self.freeze_layers:
                if layer_idx == "e":
                    modules.append(model.bert.embeddings)
                else:
                    modules.append(model.bert.encoder.layer[int(layer_idx)])
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = False

        # bitfit
        for n,p in model.named_parameters():
            if not("bias" in n or "model.classifier" in n):
                p.requires_grad = False
        logging.info(get_parameter_number(model))

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def _get_loss(x: torch.Tensor, t: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Compute cross-entropy loss.

    Args:
        x (torch.Tensor): Output of the model.
        t (torch.Tensor): Targets.
        num_classes (int, optional): Maximum number of classes. Defaults to 10.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    # y = clamp_probs(softmax(x))
    # logy = -torch.log(y)
    # loss = torch.mean(torch.sum(logy * F.one_hot(t, num_classes), dim=1))
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(x.view(-1, num_classes), t.view(-1))
    return loss


def get_loss(
    model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, num_classes: int = 10
) -> torch.Tensor:
    """Cross-entropy loss. Given a pytorch model, it computes the cross-entropy loss.

    Args:
        model (torch.nn.Module): PyTorch model.
        x (torch.Tensor): Input tensor for the PyTorch model.
        t (torch.Tensor): Targets.
        num_classes (int, optional): Maximum number of classes. Defaults to 10.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    y = model(x)[0]
    return _get_loss(y, t, num_classes)


def functional_get_loss(
    params: Tuple[torch.nn.Parameter, ...],
    model: Callable[[Tuple[torch.nn.Parameter, ...], torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    num_classes: int = 10,
    buffers = None
) -> torch.Tensor:
    """Functional cross-entropy loss. Given a functional version of a pytorch model, which can be obtained with
    `fmodel, params = functorch.make_functional(model)`, it computes the cross-entropy loss.

    Args:
        params (Tuple[torch.nn.Parameter, ...]): Model parameters obtained by `fmodel, params = fc.make_functional(model)`.
        model (Callable[[Tuple[torch.nn.Parameter, ...], torch.Tensor], torch.Tensor]): Functional version of a pytorch model,
            obtained by fmodel, `params = fc.make_functional(model)`
        x (torch.Tensor): Input tensor for the PyTorch model.
        t (torch.Tensor): Targets.
        num_classes (int, optional): Maximum number of classes. Defaults to 10.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    y = model(params,buffers, x)[0]
    return _get_loss(y, t, num_classes)

class TextClassificationTrainer:
    def __init__(self, args, device, model, train_dl=None, test_dl=None):
        self.args = args
        self.device = device

        # set data
        self.num_labels = args.num_labels
        self.set_data(train_dl, test_dl)

        # model
        self.model = model
        if self.args.model_type == "distilbert":
            self.model.add_module('pre_classifier',nn.Sequential())

        # training results
        self.results = {}
        self.best_accuracy = 0.0

        # freeze
        self.freeze_layers = args.freeze_layers.split(",") if args.freeze_layers else []

    def set_data(self, train_dl=None, test_dl=None):
        # Used for fedtrainer
        self.train_dl = train_dl
        self.test_dl = test_dl

    def train_model(self, device=None):
        if not device:
            device = self.device

        logging.info("train_model self.device: " + str(device))
        self.model.to(device)

        # build optimizer and scheduler
        iteration_in_total = len(
            self.train_dl) // self.args.gradient_accumulation_steps * self.args.epochs
        # optimizer, scheduler = self.build_optimizer(self.model, iteration_in_total)
        optimizer = self.build_optimizer(self.model, iteration_in_total)

        # training result
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        if self.args.fl_algorithm == "FedProx":
            global_model = copy.deepcopy(self.model)

        self.grad = [torch.zeros_like(p) for p in self.model.parameters()]

        for epoch in range(0, self.args.epochs):

            for batch_idx, batch in enumerate(self.train_dl):
                # if batch_idx == 3000:
                #     print(f"avg forward: {sum(forward)/len(forward)}")
                #     print(f"avg backward: {sum(backward)/len(backward)}")
                #     print(f"avg optimize: {sum(optimize)/len(optimize)}")
                self.model.train()
                batch = tuple(t for t in batch)
                # dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                x = batch[1].to(device)
                labels = batch[4].to(device)

                # (loss), logits, (hidden_states), (attentions)
                # t1 = time.time()
                # with autocast():
                output = self.model(x)
                
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # t2 = time.time()
                # logging.info(f"forward use time: {t2-t1}")
                # forward.append(t2-t1)

                if self.args.fl_algorithm == "FedProx":
                    fed_prox_reg = 0.0
                    mu = self.args.fedprox_mu
                    for (p, g_p) in zip(self.model.parameters(),
                                        global_model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((p - g_p.data)) ** 2)
                    loss += fed_prox_reg

                # model outputs are always tuple in pytorch-transformers (see doc)
                # loss = outputs[0]
                # logging.info(loss)
                current_loss = loss.item()
                logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
                                                                           len(self.train_dl), current_loss))

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                # t1 = time.time()
                loss.backward()
                for i,p in enumerate(self.model.parameters()):
                    if p.grad != None:
                        self.grad[i] += copy.deepcopy(p.grad.data)
                self.model.zero_grad()

                
                if self.args.is_debug_mode == 1 and global_step > 3:
                    break
        # results, _, _ = self.eval_model(self.args.epochs-1, global_step)
        # logging.info(results)
        return global_step, tr_loss
        # return global_step, tr_loss / global_step

    def eval_model(self, epoch=0, global_step=0, device=None):
        if not device:
            device = self.device

        results = {}

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(self.test_dl)
        test_sample_len = len(self.test_dl.dataset)
        preds = np.empty((test_sample_len, self.num_labels))

        out_label_ids = np.empty(test_sample_len)
        self.model.to(device)
        self.model.eval()
        logging.info("len(test_dl) = %d, n_batches = %d" % (len(self.test_dl), n_batches))
        for i, batch in enumerate(self.test_dl):
            with torch.no_grad():
                batch = tuple(t.to(device) for t in batch)

                x = batch[1]
                labels = batch[4]

                output = self.model(x)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()
                # logging.info("test. batch index = %d, loss = %s" % (i, str(eval_loss)))

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else test_sample_len
            # logging.info("batch index = %d, start_index = %d, end_index = %d" % (i, start_index, end_index))
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps

        model_outputs = preds
        preds = np.argmax(preds, axis=1)

        result, wrong = self.compute_metrics(preds, out_label_ids, self.test_dl.examples)
        result["eval_loss"] = eval_loss
        results.update(result)


        self.results.update(result)
        logging.info(self.results)

        return result, model_outputs, wrong

    def compute_metrics(self, preds, labels, eval_examples=None):
        assert len(preds) == len(labels)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )

    def build_optimizer(self, model, iteration_in_total):
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        self.args.warmup_steps = warmup_steps if self.args.warmup_steps == 0 else self.args.warmup_steps
        logging.info("warmup steps = %d" % self.args.warmup_steps)
        # freeze exps only apply for distilbert
        # if self.args.model_type == "distilbert" or self.args.model_type == "bert":
        self.freeze_model_parameters(model)
        if self.args.fl_algorithm == "FedOPT" or self.args.fl_algorithm == "":
            optimizer = AdamW(model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        else:
            optimizer = SGD(model.parameters(), lr=self.args.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=iteration_in_total
        # )
        # return optimizer, scheduler
        return optimizer
    
    def freeze_model_parameters(self, model):
        modules = list()
        logging.info("freeze layers: %s" % str(self.freeze_layers))
        if self.args.model_type == "distilbert":
            for layer_idx in self.freeze_layers:
                if layer_idx == "e":
                    modules.append(model.distilbert.embeddings)
                else:
                    modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        elif self.args.model_type == "bert":
            for layer_idx in self.freeze_layers:
                if layer_idx == "e":
                    modules.append(model.bert.embeddings)
                else:
                    modules.append(model.bert.encoder.layer[int(layer_idx)])
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = False

        # # bitfit
        # for n,p in model.named_parameters():
        #     if not("bias" in n or "classifier" in n):
        #         p.requires_grad = False
        logging.info(get_parameter_number(model))