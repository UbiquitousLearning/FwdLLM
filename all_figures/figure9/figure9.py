# figure 9
import matplotlib.pyplot as plt

linestyle = [(0, ( )), (0, (3, 1,1,1,1,1)), (0, (4, 1,1,1)), (0, (1, 1)),(0, (2, 4)), (0, (5, 1)),(0,(3,1,3,1)),(0,(4,3,3,1))]*100
colors=['r','g','b','y','c','m','k',"tan"] * 100
lw = 3

var_list = []
i = 2
with open("./fedFwd_distilbert_agnews_lr0.01_client_num_100_numerical_var_threthod_0.1.log","r") as f:
    lines = f.readlines()
    for line in lines:
        if 'num of fwdgrad: 100, var:' in line:
            var = float(line[line.rfind(':')+1:])
            var_list.append(var)
var_list = var_list[::15]
plt.plot([i*15 for i in list(range(len(var_list)))], var_list, linestyle=linestyle[i], color=colors[i],label='var',linewidth=lw)
plt.xlabel("step",fontsize=20)
plt.ylabel("var.",fontsize=20)
plt.xticks(size = 20)
plt.xlim(0,2500)
plt.yticks(size = 20)
# plt.ylim(0,0.98)
plt.legend(fontsize=8)
plt.title(f"var",fontsize=30)
# plt.show()
plt.savefig("./figure9")