conda create -n forwardfl python=3.7
conda activate forwardfl

pip3 install --no-cache-dir --target=/data2/wyz/forwardfl/lib/pytho
n3.7/site-packages/ torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download
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

cd FedML; git submodule init; git submodule update; cd ../; 

download dataset

.pytorch.org/whl/cu113/torch_stable.html
## 前向梯度实验，模型: DistilBERT, 数据集: AGNEWS。
```python
conda activate fwdgrad_py38
cd experiments/distributed/transformer_exps/run_tc_exps
sh run_text_classification.sh 1000 0.01 FedFwd
```
