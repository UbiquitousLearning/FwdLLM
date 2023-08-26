import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.figure(figsize=(7,4),dpi=300)
data = defaultdict(list)

# layer = "distilbert.transformer.layer.0.output_adapters.adapters.rotten tomato.adapter_up.weight"
# layer = "distilbert.transformer.layer.0.output_adapters.adapters.rotten tomato.adapter_down.0.weight"
layer = "distilbert.transformer.layer.4.output_adapters.adapters.rotten tomato.adapter_up.bias"
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
plt.plot([float(k[4:-1]) for k in k_list],y,linewidth=5)
plt.errorbar([float(k[4:-1]) for k in k_list],y,yerr,fmt='o',ecolor='r',color='black',elinewidth=5,capsize=10,markersize=10)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xscale('log')
#fmt :   'o' ',' '.' 'x' '+' 'v' '^' '<' '>' 's' 'd' 'p'
# plt.tital("figure 3b")
plt.xlabel("#Perturbation",fontsize=30)
plt.ylabel("Cosine Similarity   ",fontsize=30)
# plt.show()
plt.savefig("/data/wyz/ForwardFL-Latex/figs/design-forward-gradient-similarity_wyz.pdf", bbox_inches="tight")
