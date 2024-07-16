import pandas as pd

# 提取自变量和因变量
data_set = pd.read_csv('data1.csv');
x = data_set.iloc[:, [2,3]].values
y = data_set.iloc[:, 4].values

# 分割测试集和训练集
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# 将数据标准化
from sklearn.preprocessing import StandardScaler    
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# 开始训练
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier( n_estimators=10,criterion='entropy')
classifier.fit(x_train, y_train)

# 开始预测
y_pred = classifier.predict(x_test)  
y_pred

# 混淆矩阵验证准确性
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

#### =================求P值============== ###
# 预测一个样本
prediction = classifier.predict_proba(x_test)  # 返回样本属于每个类别的概率
 
# 获取样本的类别概率
class0_prob = prediction[0, 0]
class1_prob = prediction[0, 1]
 
# 计算近似p值
p_value = class0_prob / (class0_prob + class1_prob)

print(f"The approximate p-value for the sample is: {p_value}")
