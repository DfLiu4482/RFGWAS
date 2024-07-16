import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt


y = pd.read_csv('ttz/imf_phenotype_figshare.csv')
y = y.iloc[:,1].values
x = pd.read_csv('ttz/Neogen_China_POR80KV01_20170813.csv')
feature_names = x.iloc[:,0].values
x = x.iloc[:,1:].values

x = x.T



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#stdsc = StandardScaler()
#x_train = stdsc.fit_transform(x_train)
# 将测试数据标准化
#x_test = stdsc.transform(x_test)

forest = RandomForestRegressor(random_state=0, n_estimators=200)
forest.fit(x_train, y_train)

#预测
y_pred = forest.predict(x_test)
score =  forest.score(x_test,y_test)
#print("y_test",y_test)
#print("y_pred",y_pred)
print("score:", score)

# 获取特征重要性
feature_importances = forest.feature_importances_
# 打印特征重要性
print("Feature importances:", feature_importances)

sorted_indices = np.argsort(feature_importances)[::-1]
 
sorted_indices = np.argsort(feature_importances)[::-1]
 
# 绘制特征的重要性条形图
plt.barh(feature_names[sorted_indices[:10]], feature_importances[sorted_indices[:10]])
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Feature Importance Plot")
 
# 显示图形
plt.show()


# 评估模型-越小越准确均方根误差（RMSE）来衡量模型的表现。这个指标越小，说明模型预测越准确。
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
#print(f"RMSE: {rmse}")

# 计算评价指标
#mse = mean_squared_error(y_test, y_pred)
#rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
 
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R^2: {r2}")


#交叉验证
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rf_s = cross_val_score(forest, x, y, cv=10)
plt.plot(range(1,11), rf_s, label='RF')
plt.legend()
plt.show()