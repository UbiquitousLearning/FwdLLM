import logging
import os
from sre_parse import GLOBAL_FLAGS
import sys
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process,dequantize_params,quantize_params

class FedAVGClientManager(ClientManager):
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

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)
        
        # global_model_params = dequantize_params(global_model_params)
        self.round_idx = 0
        weight_list, sample_list = [], []
        for id in client_index:
            self.trainer.update_model(copy.deepcopy(global_model_params))
            self.trainer.update_dataset(id)
            logging.info("#######training########### round_id = %d" % self.round_idx)
            weights, local_sample_num = self.trainer.train(self.round_idx)
            self.send_model_to_server(0, copy.deepcopy(weights), local_sample_num)
            # weight_list.append(copy.deepcopy(weights))
            # sample_list.append(local_sample_num)
        self.trainer.trainer.model_trainer.scheduler.step()
        # self.send_model_to_server(0, weight_list, sample_list)

            # self.__train()

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)
        if self.args.use_quantize:
            model_params = dequantize_params(model_params)
            # for k in model_params.keys():
            #     model_params[k] = self.compressor.decompress(model_params[k][0],model_params[k][1])

        # self.trainer.update_model(model_params)
        # self.trainer.update_dataset(int(client_index))
        # self.round_idx += 1
        # self.__train()
        self.round_idx += 1
        weight_list, sample_list = [], []
        for id in client_index:
            self.trainer.update_model(model_params)
            self.trainer.update_dataset(id)
            logging.info("#######training########### round_id = %d" % self.round_idx)
            weights, local_sample_num = self.trainer.train(self.round_idx)
            self.send_model_to_server(0, copy.deepcopy(weights), local_sample_num)
            # weight_list.append(copy.deepcopy(weights))
            # sample_list.append(local_sample_num)
        self.trainer.trainer.model_trainer.scheduler.step()
        # self.send_model_to_server(0, weight_list, sample_list)
        if self.round_idx == self.num_rounds - 1:
            post_complete_message_to_sweep_process(self.args)
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        weights, local_sample_num = self.trainer.train(self.round_idx)
        self.send_model_to_server(0, weights, local_sample_num)
