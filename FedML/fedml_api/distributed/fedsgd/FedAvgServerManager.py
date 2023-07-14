from distutils.ccompiler import gen_lib_options
import logging
import os, signal
import sys
import torch
import functorch

from .message_define import MyMessage
from .utils import transform_tensor_to_list, post_complete_message_to_sweep_process,quantize_params,dequantize_params

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))
try:
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.server.server_manager import ServerManager
except ImportError:
    from FedML.fedml_core.distributed.communication.message import Message
    from FedML.fedml_core.distributed.server.server_manager import ServerManager

class FedAVGServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False, preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.accumulated_error = None

    def run(self):
        super().run()

    def send_init_msg(self):
        
        # sampling clients
        self.client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        
        # if self.args.forward_mode:
        #     global_model_params = self.aggregator.get_global_model()
        # else:
        global_model_params = self.aggregator.get_global_model_params()
        
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)

        # global_model_params = quantize_params(global_model_params)

        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, self.client_indexes[process_id - 1])

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)
        
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_GRAD_TO_SERVER,
                                              self.aggregate_tmp_grad)
        
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_VAR_TO_SERVER,
                                              self.get_var)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate(self.round_idx)
            # if self.round_idx%10==0:
            #     torch.save(self.aggregator.trainer.model, "/home/wuyaozong/agnews_model.pth")
            if self.aggregator.var > self.aggregator.var_threthod:
                for receiver_id in range(1, self.size):
                    self.send_message_cal_more_grad(receiver_id)
            else:
                self.aggregator.test_on_server_for_all_clients(self.round_idx)

                # start the next round
                self.round_idx += 1
                if self.round_idx == self.round_num-1:
                    post_complete_message_to_sweep_process(self.args)
                    self.finish()
                    return
                if self.is_preprocessed:
                    if self.preprocessed_client_lists is None:
                        # sampling has already been done in data preprocessor
                        self.client_indexes = [self.round_idx] * self.args.client_num_per_round
                    else:
                        self.client_indexes = self.preprocessed_client_lists[self.round_idx]
                else:
                    # sampling clients
                    self.client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                                    self.args.client_num_per_round)
                
                print('indexes of clients: ' + str(self.client_indexes))
                print("size = %d" % self.size)
                # if self.args.is_mobile == 1:
                #     global_model_params = transform_tensor_to_list(global_model_params)
                # if self.args.use_quantize:
                #     global_model_params,self.accumulated_error = quantize_params(global_model_params,self.accumulated_error)
                    # for k in global_model_params.keys():
                    #     global_model_params[k] = self.compressor.compress(global_model_params[k])

                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params,
                                                        self.client_indexes[receiver_id - 1])
                
    def aggregate_tmp_grad(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate(self.round_idx)

            if self.aggregator.var > self.aggregator.var_threthod:
                if self.aggregator.var > self.aggregator.var_threthod:
                    for receiver_id in range(1, self.size):
                        self.send_message_cal_more_grad(receiver_id)
            else:
                for receiver_id in range(1, self.size):
                    self.send_message_aggregate_grad_to_client(receiver_id, global_model_params,
                                                        self.client_indexes[receiver_id - 1])

    # 理论上来说应该是在server端计算var，但用mpi传大量fwdgrad不知道会不会慢，目前先在client端算，直接传var 
    def get_var(self,msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        var = msg_params.get("var")

        self.aggregator.var = var



    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, client_index)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, client_index)
        self.send_message(message)

    def send_message_aggregate_grad_to_client(self, receive_id, global_model_params, client_index):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SEND_GRAD_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, client_index)
        self.send_message(message)

    def send_message_cal_more_grad(self, receive_id):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_MORE_V, self.get_sender_id(), receive_id)
        self.send_message(message)
