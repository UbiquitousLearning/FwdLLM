# FwdLLM: Federated Fine-tuning of Billion-Sized Language Models with Forward Gradient

FwdLLM is the first federated learning framework for large language model (LLM) without backpropagation.
We successfully enables federated fine-tuning of LLaMA-7b, with only 1.5GB peak memory usage on mobile devices.

Paper link: https://arxiv.org/abs/2308.13894

## TODO:
- [ ] 如果环境不好装的话，提供docker
- [ ] future work: 补充llama的实验代码

## Installation
<!-- http://doc.fedml.ai/#/installation -->
After `git clone`-ing this repository, please run the following command to install our dependencies.

```bash
conda create --name fwdgrad python=3.7.15
conda activate fwdgrad
pip3 install cpython
pip install -r requirements.txt
# some nail wheels, we install them manually
conda install mpi4py=3.0.3=py37hf046da1_1
conda install six==1.15.0

cd FedML; git submodule init; git submodule update; cd ../; 
```

## Code Structure of FwdLLM

- `FedML`: a soft repository link generated using `git submodule add https://github.com/FedML-AI/FedML`.

- `data`: provide data downloading scripts and raw data loader to process original data and generate h5py files. Besides, `data/advanced_partition` offers some practical partition functions to split data for each client.

- `data_preprocessing`: preprocessors, examples and utility functions for each task formulation.

- `data_manager`: data manager is responsible for loading dataset and partition data from h5py files and driving preprocessor to transform data to features.

- `model`: advanced NLP models. You can define your own models in this folder.

- `trainer`: please define your own `trainer.py` by inheriting the base class in `FedML/fedml-core/trainer/fedavg_trainer.py`.
Some tasks can share the same trainer.

- `experiments`: 
    1. `experiments/distributed/transformer_exps` is the entry point for federated training. It contains experiments for different tasks. We start from `experiments/distributed/transformer_exps/run_tc_exps`.
    2. `experiments/centralized` is used to get the reference model accuracy for FL. 

- `forward_training`: The forward trainer that uses forward gradient to optimize model parameters. utils includes some tools for forward gradient calculation, such as `calculate_jvp` to calculation Jacobian-vector product using numerical differentiation, `calculate_var` to calculate the variance between multiple forward gradients, `calculate_cos_sim` to calculate the cosine similarity of the perturbation to the previous round gradient.

## Data Preparation
We have pre-processed four datasets including AGNEWS, YAHOO, YELP-P and Squad.(Need network access to drive.google.com)
```bash
gdown https://drive.google.com/uc?id=10S3Zg9HFmBuDkOusycefkugOCu27s0JT
tar -zxvf fednlp_data.tar
```

## Demo Experiments: AGNEWS for DistilBERT (Discriminative)
## 前向梯度实验，模型: DistilBERT, 数据集: AGNEWS。
```python
conda activate fwdgrad
cd experiments/distributed/transformer_exps/run_tc_exps
sh run_text_classification.sh 100 0.01 FedFwd
```

## Results
The training log will be saved in `ForwardFL/experiments/distributed/transformer_exps/run_tc_exps/log/new/`
You can find the the accuracy changes of the model by searching for `acc`.

Alternatively, you can run the following command to print the model's acc:
```bash
grep "'acc':" log/new/test_fedFwd_distilbert_agnews_lr0.01_client_num_100_numerical.log
```
The following results were obtained on 45 GB NVIDIA A40 :

`57993 2023-12-29,11:36:41.934 - {tc_transformer_trainer_distribute.py (208)} - eval_model(): {'mcc': 0.21389454684597403, 'tp': 558, 'tn': 413, 'fp': 97, 'fn': 93, 'acc': 0.3932894736842105, 'eval_loss': 1.3563928922853972}
57993 2023-12-29,11:44:24.629 - {tc_transformer_trainer_distribute.py (208)} - eval_model(): {'mcc': 0.43467640667443247, 'tp': 1405, 'tn': 711, 'fp': 297, 'fn': 34, 'acc': 0.5664473684210526, 'eval_loss': 1.3242651747402392}
57993 2023-12-29,11:51:29.268 - {tc_transformer_trainer_distribute.py (208)} - eval_model(): {'mcc': 0.5526736797408484, 'tp': 959, 'tn': 1682, 'fp': 54, 'fn': 478, 'acc': 0.655921052631579, 'eval_loss': 1.2980173840020832}
57993 2023-12-29,11:59:11.228 - {tc_transformer_trainer_distribute.py (208)} - eval_model(): {'mcc': 0.6220521689647075, 'tp': 1078, 'tn': 1316, 'fp': 96, 'fn': 137, 'acc': 0.7147368421052631, 'eval_loss': 1.2667237215293081}
57993 2023-12-29,12:06:38.357 - {tc_transformer_trainer_distribute.py (208)} - eval_model(): {'mcc': 0.643593099979723, 'tp': 1450, 'tn': 1497, 'fp': 173, 'fn': 177, 'acc': 0.7261842105263158, 'eval_loss': 1.2404077385601244}
57993 2023-12-29,12:14:13.438 - {tc_transformer_trainer_distribute.py (208)} - eval_model(): {'mcc': 0.6495114313174073, 'tp': 1570, 'tn': 1556, 'fp': 195, 'fn': 185, 'acc': 0.7255263157894737, 'eval_loss': 1.1989253973960876}
57993 2023-12-29,12:23:46.784 - {tc_transformer_trainer_distribute.py (208)} - eval_model(): {'mcc': 0.6794642083719067, 'tp': 1419, 'tn': 1638, 'fp': 114, 'fn': 271, 'acc': 0.7543421052631579, 'eval_loss': 1.1541506526972118}
57993 2023-12-29,12:34:21.890 - {tc_transformer_trainer_distribute.py (208)} - eval_model(): {'mcc': 0.7215069617259151, 'tp': 1161, 'tn': 1642, 'fp': 69, 'fn': 157, 'acc': 0.7896052631578947, 'eval_loss': 1.10180882002178}
57993 2023-12-29,12:46:09.503 - {tc_transformer_trainer_distribute.py (208)} - eval_model(): {'mcc': 0.7216381315733102, 'tp': 1337, 'tn': 1660, 'fp': 69, 'fn': 229, 'acc': 0.7893421052631578, 'eval_loss': 1.04565118105788}`

<!-- ## Demo Experiments: LLaMA for Squad (Generative)

**TODO**: add the code for LLaMA-7b
### Training
```python
```

### Evaluation（给别人我们训好的checkpoint）
```python
``` -->


## Citation
Please cite our FwdLLM paper if it helps your research.
```bib
@misc{xu2023federated,
      title={Federated Fine-tuning of Billion-Sized Language Models across Mobile Devices}, 
      author={Mengwei Xu and Yaozong Wu and Dongqi Cai and Xiang Li and Shangguang Wang},
      year={2023},
      eprint={2308.13894},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```