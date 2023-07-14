import copy
import logging
import random
import time
import math

import numpy as np
import torch
import wandb

from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class FedAVGAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.warmup_rounds = math.ceil(self.args.comm_round * self.args.warmup_ratio)

        # 之前的v不够，暂存在cached_v
        self.cached_v = []
        # 模型之前的准确率
        self.previous_acc = -1
        self.previous_loss = 100

        # adam optimizer
        self.optimizer = torch.optim.Adam(self.trainer.model.parameters(),lr=self.args.learning_rate)
        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.comm_round
        # )

    def get_global_model_params(self):
        return self.trainer.get_model_params()
    
    def get_global_model(self):
        return self.trainer.get_model()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self,current_round):
        start_time = time.time()
        model_list = []
        training_num = 0

        # self.warmup_rounds = 20
        # if current_round < self.warmup_rounds:
        #     ratio = float(current_round+1) / float(max(1, self.warmup_rounds))
        # else:
        #     ratio = max(
        #     0.0, float(self.args.comm_round - current_round) / float(max(1, self.args.comm_round - self.warmup_rounds))
        # )
        # learning_rate = self.args.learning_rate * ratio
        # logging.info(f"learning rate: {learning_rate}")

        for idx in range(self.worker_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]
        
        # self.model_dict在聚合的过程中会被改变,很奇怪，这里先存一个deepcopy吧，用于后面cache_v
        # model_dict_cached = copy.deepcopy(self.model_dict)
        # origin_param = copy.deepcopy(self.get_global_model_params())

        # cached_v:  (num, params)
        logging.info(f"len of cached v: {len(self.cached_v)}")
        for cached_v in self.cached_v:
            model_list.append(cached_v)
            training_num += cached_v[0]
        logging.info(f"training_num : {training_num}")

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))
        
        # old_param = self.get_global_model_params()
        old_param = self.trainer.model.parameters()
        
        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for id,k in enumerate(averaged_params):
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                # w = local_sample_number / training_num
                if i == 0:
                    averaged_params[id] = local_model_params[id]
                else:
                    averaged_params[id] += local_model_params[id]
            # logging.info(averaged_params[id].device)
            # logging.info(self.args.learning_rate.device)
            # logging.info(training_num .device)
            # old_param[k] -= learning_rate * averaged_params[id] / training_num 
                # if id == 22:
                #     logging.info(sum(local_model_params[id]))
            p = next(old_param).to("cpu")
            g = (averaged_params[id] / training_num)  
            p.grad = g  
        self.optimizer.step()
        # self.scheduler.step()
        self.optimizer.zero_grad() 

        # 在前1000个batch上快速测试一下新模型的准确率
        # cur_model_acc = self.trainer.quick_test(self.device,500)
        # if cur_model_acc>self.previous_acc or len(self.cached_v) >= 9:
        #     # 当前模型更好,清空cached_v,更新cur_acc
        #     self.cached_v = []
        #     self.previous_acc = cur_model_acc
        # cur_model_loss = self.trainer.quick_test_loss(self.device,500)
        # if cur_model_loss<self.previous_loss:
        #     # 当前模型更好,清空cached_v,更新cur_acc
        #     self.cached_v = []
        #     self.previous_loss = cur_model_loss
        if True:
            pass
        else:
            logging.info("current model is not good enough, calculate more v")
            # 当前模型不行，v不够，暂存起来，后面再计算更多的v
            for idx in range(self.worker_num):
                self.cached_v.append((self.sample_num_dict[idx], model_dict_cached[idx]))
            # 模型改回去
            self.set_global_model_params(origin_param)

        old_param = self.get_global_model_params()

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return old_param

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        index_list = [[] for _ in range(self.args.worker_num)]

        for i in range(len(client_indexes)):
            index_list[i%self.args.worker_num].append(client_indexes[i])

        client_indexes = index_list

        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num  = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            if self.trainer.test_on_the_server(self.train_data_local_dict, self.test_data_local_dict, self.device, self.args):
                return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            for client_idx in range(self.args.client_num_in_total):
                # train data
                metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
                train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            # wandb.log({"Train/Acc": train_acc, "round": round_idx})
            # wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)
                
            test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
                'test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)
