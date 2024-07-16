#模拟数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###################模拟基因型数据##################

# 设置随机种子以获得可重复的结果
np.random.seed(123)

# 模拟参数
n_samples = 1000  # 个体数量
n_markers = 10000  # 标记点（SNP）数量
maf = 0.2  # 最常见等位基因频率

# 生成基因型频率
genotype_freqs = [maf**2, 2*maf*(1-maf), (1-maf)**2]  # AA, Aa, aa 的频率

# 模拟基因型数据
# 我们假设所有SNP都是独立的，并且每个SNP的等位基因频率相同
genotype_data = np.random.choice([0, 1, 2], size=(n_samples, n_markers), p=genotype_freqs)

# 将基因型数据转换为DataFrame
genotype_df = pd.DataFrame(genotype_data, columns=[f'SNP{i+1}' for i in range(n_markers)])

# 显示前5行
print(genotype_df.head())

###################模拟基因型数据##################

###################模拟表型数据##################

# 假设前10个SNP与表型相关，每个SNP的效应大小是随机的
effect_sizes = np.random.randn(10)

# 生成与SNP相关的表型数据
phenotype_related = np.dot(genotype_df.iloc[:, :10], effect_sizes)

# 添加随机误差以模拟其他未知因素和环境影响
error = np.random.randn(n_samples) * 0.5

# 生成最终的表型数据
phenotype_data = phenotype_related + error

# 将表型数据转换为DataFrame
phenotype_df = pd.DataFrame({'Phenotype': phenotype_data})

# 显示表型数据
print(phenotype_df.head())

###################模拟表型数据##################

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

x_train, x_test, y_train, y_test = train_test_split(genotype_df, phenotype_df, test_size=0.25, random_state=0)
forest = RandomForestRegressor(random_state=0, n_estimators=200)
forest.fit(x_train, y_train)
score =  forest.score(x_test,y_test)
print("score:", score)

# 评估模型-越小越准确均方根误差（RMSE）来衡量模型的表现。这个指标越小，说明模型预测越准确。
y_pred = forest.predict(x_test)
# 计算评价指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
 
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R^2: {r2}")

#交叉验证
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rf_s = cross_val_score(forest, genotype_df, phenotype_df, cv=10)
plt.plot(range(1,11), rf_s, label='RF')
plt.legend()
plt.show()

# 学习曲线--调n_estimators
superpa = []
for i in range(200):
    rfc = RandomForestRegressor(random_state=0, n_estimators=i+1,n_jobs=-1)
    rfc_s = cross_val_score(rfc,genotype_df,phenotype_df,cv=10).mean()
    superpa.append(rfc_s)
print(max(superpa),superpa.index(max(superpa))+1)
plt.figure(figsize=[20,5])
plt.plot(range(1,201),superpa)
plt.show()

# 网格搜索
from sklearn.model_selection import GridSearchCV

#调max_depth
param_grid =  {'max_depth':np.arange(1,20,1)}
rfc = RandomForestRegressor(n_estimators=39, random_state=90)
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(genotype_df, phenotype_df)
GS.best_params._ #显示调整出来的最佳参数
GS.best_score #返回调整好的最佳参数对应的准确率

#调max_features
param_grid = {'max_features':np.arange(5,30,1)}
#max_features是唯一一个即能够将模型往左（低方差高偏差）推，也能够将模型往右（高方差低偏差）推的参数。我
#们需要根据调参前，模型所在的位置（在泛化误差最低点的左边还是右边）米决定我们要将ma×features往哪边调。
#现在模型位于图像左侧，我们需要的是更高的复杂度，因此我们应该把max_features往更大的方向调整，可用的特征
#越多，模型才会越复朵。max features的默认最小值是sqrt(n features),因此我们使用这个值作为调参范围的
#最小值。
rfc = RandomForestRegressor(n_estimators=39,random_state=90)
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(x_train, y_train)
GS.best_params_
GS.best_score_


#调min_samples_leaf
param_grid={'min_samples_leaf':np.arange(2,2+20,1)}
rfc = RandomForestRegressor(n_estimators=39,random_state=90)
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(x_train, y_train)
GS.best_params_
GS.best_sqore_

#调min_samples_split
param_grid={'min_samples_split':np.arange(2,2+20,1)}
rfc = RandomForestRegressor(n_estimators=39,random_state=90)
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(x_train, y_train)
GS.best_params_
GS.best_sqore_


                             
