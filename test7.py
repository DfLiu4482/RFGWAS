#GBLUP
import numpy as np
from gblup import gblup
 
# 假设有3个个体，每个个体有2个标记
# 假设数据是SNP的等位基因频率（0代表参考等位基因，1代表第一个等位基因，2代表第二个等位基因）
genotype_data = np.array([
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2]
])
 
# 计算GBLUP值
num_markers = genotype_data.shape[1]  # 标记数
num_individuals = genotype_data.shape[0]  # 个体数
 
# 假设有一个环境因子（例如，环境因子可以是农作物的生长条件）
environment_data = np.array([1.0, 2.0, 3.0])
 
# 调用GBLUP函数
blups, res_cov = gblup(genotype_data, environment_data)
 
# blups包含GBLUP值，res_cov是残差协方差矩阵
print("GBLUP values:", blups)