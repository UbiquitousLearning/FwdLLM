import logging
import os
from sre_parse import GLOBAL_FLAGS
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import post_complete_message_to_sweep_process, grad_aggregete

class FedSGDClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SEND_GRAD_TO_CLIENT,
                                              self.handle_message_receive_aggregated_grad_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_MORE_V,
                                              self.calculate_more_v)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        # ad_hoc
        # self.trainer.trainer.model_trainer.cur_v_num_index += 1

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(client_index)
        self.round_idx = 0
        # self.__train()
        self.data_id = 0
        self.train_with_data_id()

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        # ad_hoc
        # self.trainer.trainer.model_trainer.cur_v_num_index += 1
        # 方差足够小，清空暂存的fwdgrad
        if self.args.var_control:
            if self.args.perturbation_sampling:
                if self.data_id % 2:
                    self.trainer.trainer.model_trainer.old_grad = grad_aggregete(self.trainer.trainer.model_trainer.grad_pool)
                else:
                    self.trainer.trainer.model_trainer.old_grad = None
                self.trainer.trainer.model_trainer.grad_pool = []
            self.trainer.trainer.model_trainer.grad_for_var_check_list = []


        self.trainer.update_model(model_params)
        self.trainer.update_dataset(client_index)
        self.round_idx += 1

        self.data_id = 0
        self.train_with_data_id()

        # self.__train()
        if self.round_idx == self.num_rounds - 1:
            post_complete_message_to_sweep_process(self.args)
            self.finish()

    def handle_message_receive_aggregated_grad_from_server(self, msg_params):
        logging.info("handle_message_receive_aggregated_grad_from_server")
        
        # 方差足够小，清空暂存的fwdgrad
        if self.args.var_control:
            if self.args.perturbation_sampling:
                if self.data_id % 2:
                    self.trainer.trainer.model_trainer.old_grad = grad_aggregete(self.trainer.trainer.model_trainer.grad_pool)
                else:
                    self.trainer.trainer.model_trainer.old_grad = None
                self.trainer.trainer.model_trainer.grad_pool = []
            self.trainer.trainer.model_trainer.grad_for_var_check_list = []

        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.update_model(model_params)
        self.train_with_data_id()

    # 方差太大，计算更多v
    def calculate_more_v(self,msg_params):
        self.data_id -= 1
        logging.info("calculate more v")
        self.train_with_data_id()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def send_grad_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_GRAD_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def send_var_to_server(self, receive_id, var):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_VAR_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params("var", var)
        self.send_message(message)

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        weights, local_sample_num = self.trainer.train(self.round_idx)
        self.send_model_to_server(0, weights, local_sample_num)

    def train_with_data_id(self):
        logging.info("#######training########### round_id = %d data_id = %d" % (self.round_idx, self.data_id))
        
        weights, client_num = self.trainer.train_with_data_id(self.round_idx,self.data_id)
        if self.args.var_control:
            self.send_var_to_server(0,self.trainer.trainer.model_trainer.var)
        
        self.data_id += 1
        if self.data_id == len(self.trainer.train_local_list[0]):
            self.send_model_to_server(0, weights, client_num)
        else:
            self.send_grad_to_server(0, weights, client_num)
