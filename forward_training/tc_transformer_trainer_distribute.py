# coding: utf-8

from __future__ import absolute_import, division, print_function

from hashlib import shake_128
import logging

import numpy as np
import sklearn
import torch
from torch import nn
from training.utils.text_classification_utils import *
from forward_training.utils.fwdgrad_utils import *
from torch.nn import CrossEntropyLoss
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

        # training results
        self.results = {}
        self.best_accuracy = 0.0

        # freeze
        self.freeze_layers = args.freeze_layers.split(",") if args.freeze_layers else []

        self.grad = None
        if self.args.perturbation_sampling and self.args.var_control:
            self.old_grad = None
            self.grad_pool = []


        # var control
        self.grad_for_var_check_list = []
        if self.args.model_type == "distilbert":
            self.layer_id_for_check = 20
        elif self.args.model_type == "bert":
            self.layer_id_for_check = 12
        elif self.args.model_type == "roberta-large":
            self.layer_id_for_check = 12
        elif self.args.model_type == "albert":
            self.layer_id_for_check = 22
        self.var = 0

    def set_data(self, train_dl=None, test_dl=None):
        # Used for fedtrainer
        self.train_dl = train_dl
        self.test_dl = test_dl

    def train_model(self, device=None):
        if not device:
            device = self.device

        logging.info("train_model self.device: " + str(device))
        self.model.to(device)

        logging.info(get_parameter_number(self.model))
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)

        # training result
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        if self.args.perturbation_sampling:
            v_num = len(self.train_dl)
            v_buffer = {}
            index = 0
            if self.args.var_control:
                self.grad = self.old_grad
            for k,v in self.model.named_parameters():
                # logging.info(index)
                if self.grad != None and v.requires_grad:
                    # logging.info("generate v")
                    shape = v.shape
                    candidate_v = torch.randn((v_num*10,*shape),device="cpu")
                    target_grad = self.grad[index]

                    # logging.info("flatten")
                    target_grad = torch.flatten(target_grad)
                    candidate_v = torch.flatten(candidate_v,start_dim=1)

                    cos_sim = calculate_cos_sim(candidate_v,target_grad,device)
                    sorted_values, sorted_indices = torch.sort(cos_sim, descending=True)

                    v_buffer[index] = [candidate_v[i].reshape(v.shape) for i in sorted_indices[:v_num]]
                index += 1

        self.grad = [torch.zeros_like(p) for p in self.params]

        with torch.no_grad():
            for epoch in range(0, self.args.epochs):
                for batch_idx, batch in enumerate(self.train_dl):

                    batch = tuple(t for t in batch)
                    x = batch[1].to(device)
                    labels = batch[4].to(device)

                    # 优化函数
                    f = partial(
                        functional_get_loss,
                        model=self.fmodel,
                        buffers = self.buffers,
                        num_classes = self.num_labels,
                        x=x,
                        t=labels,
                    )

                    # 生成扰动
                    if self.args.perturbation_sampling and v_buffer != {}:
                        v_params = tuple([v_buffer[i][batch_idx].to(device) if p.requires_grad == True else torch.zeros_like(p) for i,p in enumerate(self.params)])
                    else:
                        v_params = tuple([torch.randn_like(p) if p.requires_grad == True else torch.zeros_like(p) for p in self.params])
                    
                    # 计算方向导数
                    loss, jvp = calculate_jvp(f, self.params, v_params)
                    
                    # 计算梯度
                    for j, fg in enumerate(self.grad):
                        fg.add_(jvp*v_params[j])
                        if self.args.var_control and j == self.layer_id_for_check:
                            self.grad_for_var_check_list.append(jvp*v_params[j])


                    current_loss = loss.item()
                    logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
                                                                            len(self.train_dl), current_loss))

                    global_step += 1
                    if self.args.evaluate_during_training and (self.args.evaluate_during_training_steps > 0
                                                                and global_step!=0  and global_step % self.args.evaluate_during_training_steps == 0):
                        results, _, _ = self.eval_model(epoch, global_step)

                    if self.args.is_debug_mode == 1 and global_step > 3:
                        break

        if self.args.var_control:
            self.var = calculate_var(self.grad_for_var_check_list,)
            logging.info(f"num of fwdgrad: {len(self.grad_for_var_check_list)}, var: {self.var}")
            if self.args.perturbation_sampling:
                self.grad_pool.append(self.grad)
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
                x = batch[1]
                labels = batch[4]

                output = self.model(x)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else test_sample_len
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

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
