import argparse
import logging
import os
import sys

import torch
from torch import nn
# this is a temporal import, we will refactor FedML as a package installation
# import wandb

# wandb.init(mode="disabled")

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from data_preprocessing.text_classification_preprocessor import TLMPreprocessor
from data_manager.text_classification_data_manager import TextClassificationDataManager

from model.transformer.model_args import ClassificationArgs

from training.tc_transformer_trainer import TextClassificationTrainer

from experiments.centralized.transformer_exps.initializer import set_seed, add_centralized_args, create_model

if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    parser = add_centralized_args(parser)  # add general args.
    # TODO: you can add customized args here.
    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    set_seed(args.manual_seed)

    # device
    device = torch.device("cpu")

    # # initialize the wandb machine learning experimental tracking platform (https://wandb.ai/automl/fednlp).
    # wandb.init(project="fednlp", entity="automl", name="FedNLP-Centralized" +
    #                                             "-TC-" + str(args.dataset) + "-" + str(args.model_name) + "-freeze-" + args.freeze_layers if args.freeze_layers else "",
    #     config=args)

    # attributes
    attributes = TextClassificationDataManager.load_attributes(args.data_file_path)
    num_labels = len(attributes["label_vocab"])

    # model
    model_args = ClassificationArgs()
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = num_labels
    model_args.fl_algorithm = ""
    model_args.update_from_dict({"epochs": args.epochs,
                                 "freeze_layers": args.freeze_layers,
                                 "learning_rate": args.learning_rate,
                                 "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                 "do_lower_case": args.do_lower_case,
                                 "manual_seed": args.manual_seed,
                                 "reprocess_input_data": args.reprocess_input_data,  # for ignoring the cache features.
                                 "overwrite_output_dir": True,
                                 "max_seq_length": args.max_seq_length,
                                 "train_batch_size": args.train_batch_size,
                                 "eval_batch_size": args.eval_batch_size,
                                 "evaluate_during_training_steps": args.evaluate_during_training_steps,
                                 "fp16": args.fp16,
                                 "data_file_path": args.data_file_path,
                                 "partition_file_path": args.partition_file_path,
                                 "partition_method": args.partition_method,
                                 "dataset": args.dataset,
                                 "output_dir": args.output_dir,
                                 "is_debug_mode": args.is_debug_mode,
                                 "use_quantize": args.use_quantize,
                                 "use_adapter": args.use_adapter,
                                 })

    model_args.config["num_labels"] = num_labels
    model_config, model, tokenizer = create_model(model_args, formulation="classification")

    # preprocessor
    preprocessor = TLMPreprocessor(args=model_args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer)

    # data manager
    dm = TextClassificationDataManager(args, model_args, preprocessor)

    train_dl, test_dl = dm.load_centralized_data()

    # Create a ClassificationModel and start train
    trainer = TextClassificationTrainer(model_args, device, model, train_dl, test_dl)
    # trainer.train_model()
    logging.info("=======================load model==========================")
    trainer.model = torch.load("/home/wuyaozong/agnews_model.pth").to(device)
    print(model)

    # logging.info("=======================load state_dict==========================")
    # trainer.model.load_state_dict(torch.load("/home/wuyaozong/model_state_dict.pth",map_location='cpu'))

    # logging.info("=======================eval==========================")
    # trainer.eval_model()

    logging.info("=======================quantize==========================")
    trainer.model = torch.quantization.quantize_dynamic(
                        model=trainer.model,  # 原始模型    
                        qconfig_spec={nn.Linear,
                                    nn.Embedding,
                                    nn.LayerNorm},  # 要动态量化的NN算子
                        dtype=torch.qint8) 
    logging.info("=======================eval==========================")
    trainer.eval_model()