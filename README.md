# FwdLLM: Federated Fine-tuning of Billion-Sized Language Models with Forward Gradient

FwdLLM is the first federated learning framework for large language model (LLM) without backpropagation.
We successfully enables federated fine-tuning of LLaMA-7b, with only 1.5GB peak memory usage on mobile devices.

Paper link: https://arxiv.org/abs/2308.13894

## TODO:
- [ ] 完成code structure中最后两项 
- [ ] 提供数据集下载链接
- [ ] 如果环境不好装的话，提供docker
- [ ] future work: 补充llama的实验代码

## Installation
<!-- http://doc.fedml.ai/#/installation -->
After `git clone`-ing this repository, please run the following command to install our dependencies.

```bash
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

- `DQ notes`: figs文件夹不需要， 论文还没开源，我们的数据、结果、图片和画图的脚本都不能放到开源的版本里。

## Data Preparation
We have pre-processed four datasets including AGNEWS, YAHOO, YELP-P and Squad.
```bash
download dataset # TODO: add download script
gdown https://drive.google.com/uc?id=10S3Zg9HFmBuDkOusycefkugOCu27s0JT
tar -zxvf fednlp_data.tar
```

## Demo Experiments: AGNEWS for DistilBERT (Discriminative)
## 前向梯度实验，模型: DistilBERT, 数据集: AGNEWS。
```python
conda activate fwdgrad_py38
cd experiments/distributed/transformer_exps/run_tc_exps
sh run_text_classification.sh 1000 0.01 FedFwd
```

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