import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


data = defaultdict(list)

layer = "distilbert.transformer.layer.0.output_adapters.adapters.rotten tomato.adapter_up.weight"
# layer = "distilbert.transformer.layer.0.output_adapters.adapters.rotten tomato.adapter_down.0.weight"
# layer = "distilbert.transformer.layer.0.output_adapters.adapters.rotten tomato.adapter_up.bias"
k_list = [f"k = {k}," for k in [1,10,100,1000,10000,100000,1000000]]

with open("./fedFwd_distilbert_agnews_lr0.01_client_num_1000_numerical_check_cos.log") as f:
    for line in f:
        if layer in line:
            for k in k_list:
                if k in line:
                    cosine = float(line[line.rfind("cos_sim")+10:line.rfind("v_shape")-2])
                    data[k].append(cosine)
                    break
for key in data:
    data[key] = np.array(data[key])
y = [data[key].mean() for key in data]
yerr = [data[key].std() for key in data]
# yerr = [data[key].var() for key in data]
plt.plot(range(len(y)),y)
plt.errorbar(range(len(y)),y,yerr,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
plt.xticks(range(len(y)),[k[:-1] for k in k_list],size=8)
#fmt :   'o' ',' '.' 'x' '+' 'v' '^' '<' '>' 's' 'd' 'p'
# plt.tital("figure 3b")
plt.xlabel("Num of Perturbations")
plt.ylabel("Cosine Similarity of Fwdgrad and True Grad")
# plt.show()
plt.savefig("./design-forward-gradient-similarity_wyz.png")
