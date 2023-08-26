# figure 9
import matplotlib.pyplot as plt

linestyle = [(0, ( )), (0, (3, 1,1,1,1,1)), (0, (4, 1,1,1)), (0, (1, 1)),(0, (2, 4)), (0, (5, 1)),(0,(3,1,3,1)),(0,(4,3,3,1))]*100
colors=['r','g','b','y','c','m','k',"tan"] * 100
lw = 5
plt.figure(figsize=(7,4),dpi=300)
var_list = []
i = 2
with open("./fedFwd_distilbert_agnews_lr0.01_client_num_100_numerical_var_threthod_0.1.log","r") as f:
    lines = f.readlines()
    for line in lines:
        if 'num of fwdgrad: 100, var:' in line:
            var = float(line[line.rfind(':')+1:])
            var_list.append(var)
var_list = var_list[::30]
plt.plot([i*30 for i in list(range(len(var_list)))], var_list, linestyle='-', color=colors[i],label='var',linewidth=lw)
plt.xlabel("Step",fontsize=30)
plt.ylabel("Variance",fontsize=30)
plt.xticks(fontsize = 25)
plt.xlim(0,2500)
plt.yticks(fontsize = 25)
# plt.ylim(0,0.98)
# plt.legend(fontsize=25)
# plt.title(f"var",fontsize=30)
# plt.show()
plt.savefig("/data/wyz/ForwardFL-Latex/figs/design-planning-variance_wyz.pdf", bbox_inches="tight")
# plt.savefig("./design-planning-variance_wyz.pdf", bbox_inches="tight")