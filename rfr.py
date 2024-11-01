# rfr.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

def execute(phe_path, gen_path):
    
    y = pd.read_csv(phe_path)
    y = y.iloc[:,1].values
    xxx = pd.read_csv(gen_path)
    feature_names = xxx.iloc[:,0].values
    xx = xxx.iloc[:,1:].values
    x = xx.T

    # 拆分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # 开始训练
    forest = RandomForestRegressor(random_state=0, n_estimators=200)
    forest.fit(x_train, y_train)

    # 获取特征重要性
    feature_importances = forest.feature_importances_
    
    sorted_indices = np.argsort(feature_importances)[::-1]
    snp_num = int(len(sorted_indices) * 0.1)
    
    # 输出前10%变量特征名称到csv
    top_feature = feature_names[sorted_indices[:snp_num]]
    df = pd.DataFrame(top_feature)
    df.to_csv('FeatureName.csv', index=False)

    
    genFileHandler(xxx,top_feature)

    # 绘制特征的重要性条形图
    plt.barh(feature_names[sorted_indices[:snp_num]], feature_importances[sorted_indices[:snp_num]])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.title("Feature Importance Plot")
    plt.savefig('feature.png')
    plt.close()
    

def genFileHandler(genData,snp_name):
    genData[snp_name]
    df = pd.DataFrame(genData[snp_name])
    df.to_csv('newGen.csv', index=False)
    print(2)

def pheFileHandler():
    print(3)
