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
conda create fwdgrad python=3.7.15
pip3 install cpython
pip install -r requirements.txt
# some nail wheels, we install them manually
conda install mpi4py=3.0.3=py37hf046da1_1
conda install six==1.15.0

pip3 install pandas
pip3 install scipy
pip3 install scikit-learn
pip3 install tensorboardX
pip3 install tqdm
pip3 install adapter-transformers==3.1.0
pip3 install functorch
pip3 install gdown

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
sh run_text_classification.sh 1000 0.01 FedFwd
```

## Results
The training log will be saved in `ForwardFL/experiments/distributed/transformer_exps/run_tc_exps/log/new/`
You can find the the accuracy changes of the model by searching for `acc`.

Alternatively, you can run the following command to print the model's acc:
```bash
grep "'acc':" log/new/test_fedFwd_distilbert_agnews_lr0.01_client_num_100_numerical.log
```
The following results were obtained on 45 GB NVIDIA A40 :

`56948 2023-12-13,00:37:50.641 - {tc_transformer_trainer_distribute.py (204)} - eval_model(): {'mcc': 0.1929170760254043, 'tp': 275, 'tn': 205, 'fp': 13, 'fn': 44, 'acc': 0.3788157894736842, 'eval_loss': 1.3653460335731507}
56948 2023-12-13,00:45:16.552 - {tc_transformer_trainer_distribute.py (204)} - eval_model(): {'mcc': 0.40932455654889816, 'tp': 1229, 'tn': 823, 'fp': 236, 'fn': 139, 'acc': 0.5477631578947368, 'eval_loss': 1.3373655635432193}
56948 2023-12-13,00:52:42.738 - {tc_transformer_trainer_distribute.py (204)} - eval_model(): {'mcc': 0.4718381916383034, 'tp': 780, 'tn': 888, 'fp': 61, 'fn': 98, 'acc': 0.5917105263157895, 'eval_loss': 1.3142965581542567}
56948 2023-12-13,00:59:51.217 - {tc_transformer_trainer_distribute.py (204)} - eval_model(): {'mcc': 0.5809650861410457, 'tp': 954, 'tn': 1546, 'fp': 104, 'fn': 327, 'acc': 0.6838157894736843, 'eval_loss': 1.2913910858254685}
56948 2023-12-13,01:07:15.732 - {tc_transformer_trainer_distribute.py (204)} - eval_model(): {'mcc': 0.616786890679179, 'tp': 1299, 'tn': 1461, 'fp': 197, 'fn': 241, 'acc': 0.7109210526315789, 'eval_loss': 1.2616328983557852}
56948 2023-12-13,01:15:24.855 - {tc_transformer_trainer_distribute.py (204)} - eval_model(): {'mcc': 0.6520238510675961, 'tp': 1110, 'tn': 1474, 'fp': 141, 'fn': 187, 'acc': 0.7386842105263158, 'eval_loss': 1.2299891811922976}
56948 2023-12-13,01:22:48.791 - {tc_transformer_trainer_distribute.py (204)} - eval_model(): {'mcc': 0.6773361193622469, 'tp': 1221, 'tn': 1628, 'fp': 100, 'fn': 285, 'acc': 0.7568421052631579, 'eval_loss': 1.1899380558415462}
56948 2023-12-13,01:32:01.860 - {tc_transformer_trainer_distribute.py (204)} - eval_model(): {'mcc': 0.697729266659397, 'tp': 1337, 'tn': 1645, 'fp': 120, 'fn': 270, 'acc': 0.7714473684210527, 'eval_loss': 1.1469027110149985}`

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