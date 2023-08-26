import torch
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

plt.figure(figsize=(8,6),dpi=500)

# 向量A
torch.manual_seed(42)
vector_A = torch.randn(768)

# 矩阵B，由1000个768维向量组成
matrix_B = torch.randn(10000, 768)

# 计算余弦相似度
cosine_similarities = torch.cosine_similarity(vector_A.unsqueeze(0), matrix_B, dim=1)
abs_cosine_similarities = torch.abs(cosine_similarities)

# 设置横坐标范围和间隔
start = 0.0
end = abs_cosine_similarities.max().item()
interval = 0.001

# 统计每个范围内的向量数量和余弦相似度的平均值
counts = []
means = []
bins = []

for i in range(int((end - start) / interval) + 1):
    lower = start + i * interval
    upper = start + (i + 1) * interval
    count = torch.sum((abs_cosine_similarities >= lower) & (abs_cosine_similarities < upper)).item()
    mean = torch.mean(cosine_similarities[(abs_cosine_similarities >= lower) & (abs_cosine_similarities < upper)].abs()).item()/end
    counts.append(count)
    means.append(mean)
    bins.append(lower)

# 创建颜色映射
cmap = plt.cm.get_cmap('Reds')

# 根据余弦相似度的平均值设置颜色
colors = [cmap(mean) for mean in means]

# 绘制条形图
plt.bar(bins, counts, width=interval, align='edge', color=colors)
plt.xlabel('Absolute Value\nof Cosine Similarity',fontsize=30)
plt.xlim(0,0.15)
plt.xticks([0.04,0.08,0.12],fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel('Count',fontsize=30)
# plt.title('figure 4b')
cbar = plt.colorbar(ScalarMappable(cmap=cmap))
cbar.ax.tick_params(labelsize=25)
# plt.show()
plt.savefig("/data/wyz/ForwardFL-Latex/figs/design-sampling-statistic_wyz.pdf", bbox_inches="tight")
