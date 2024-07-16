import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  

data_set = pd.read_csv('data1.csv');

# 从data_set中选择第2列和第3列的数据作为特征变量x，选择第4列的数据作为目标变量y
x = data_set.iloc[:, [2,3]].values  
y = data_set.iloc[:, 4].values

#导入了train_test_split函数，用于将数据集划分为训练集和测试集。
# 通过调用train_test_split函数，并传入特征变量x和目标变量y，以及测试集  的大小（test_size=0.25）和随机种子（random_state=0），
#  将数据集划分为x_train、x_test、y_train和y_test。

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)



from sklearn.preprocessing import StandardScaler    
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)    
x_test = sc.transform(x_test)    

data_set.head(5)

x_train[0:5]
x_test[0:5]


#首先从sklearn.ensemble模块导入RandomForestClassifier类。然后，使用RandomForestClassifier类创建了一个名为classifier的随机森林分类器对象。

# 在创建分类器对象时，通过n_estimators参数指定了随机森林中决策树的数量为10个。通过criterion参数指定了使用熵（entropy）作为决策树的评估标准。

# 接下来，使用fit()方法将训练数据x_train和对应的标签y_train传入分类器对象，以训练模型。fit()方法会根据传入的数据和标签来构建随机森林模型。

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier( n_estimators=10,criterion='entropy')  
classifier.fit(x_train, y_train)  
score = classifier.score(x_test,y_test)
print(score)

y_pred = classifier.predict(x_test)  
y_pred

from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, y_pred)
cm

#交叉验证
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rf_s = cross_val_score(classifier, x, y, cv=10)
plt.plot(range(1,11), rf_s, label='RF')
plt.legend()
plt.show()
