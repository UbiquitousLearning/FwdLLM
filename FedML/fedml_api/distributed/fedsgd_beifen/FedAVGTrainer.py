from .utils import transform_tensor_to_list,quantize_params
import logging


class FedAVGTrainer(object):

    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                 train_data_num, device, args, model_trainer):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        self.device = device
        self.args = args
        self.accumulated_error = None

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = [self.train_data_local_dict[id] for id in client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index[0]]
        self.test_local = self.test_data_local_dict[client_index[0]]

        self.train_local_list = [[data for data in self.train_local[i]] for i in range(len(self.train_local))]

    def train(self, round_idx = None):
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        if self.args.use_quantize:
            weights,self.accumulated_error = quantize_params(weights,self.accumulated_error)

        return weights, self.local_sample_number
    
    def train_with_data_id(self, round_idx = None, data_id = 0):
        self.args.round_idx = round_idx
        self.trainer.train([train_local[data_id] for train_local in self.train_local_list], self.device, self.args)

        # weights = self.trainer.get_model_params()
        weights = [para.detach().cpu() for para in self.trainer.model_trainer.grad]

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        if self.args.use_quantize:
            weights,self.accumulated_error = quantize_params(weights,self.accumulated_error)

        return weights, len(self.train_local)

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = train_metrics['test_correct'], \
                                                          train_metrics['test_total'], train_metrics['test_loss']

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = test_metrics['test_correct'], \
                                                          test_metrics['test_total'], test_metrics['test_loss']

        return train_tot_correct, train_loss, train_num_sample, test_tot_correct, test_loss, test_num_sample