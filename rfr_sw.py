# rfr.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

#窗口大小
win = 3
#取出top百分比
top = 0.1

def sliding_window(gen_path):
    # 按照窗口大小将csv文件拆分
    df = pd.read_csv(gen_path)

    merged_dfs = []

    # 计算窗口数量
    num_windows = len(df) - 1

    # 对每个窗口进行迭代
    for i in range(num_windows):
        start = i
        end = start + 3
        # 获取窗口内的行
        window_df = df.iloc[start:end]
        if len(window_df) >= win: 
            # 对'ID'列使用'#'连接
            merged_id = '#'.join(window_df["ID"].astype(str))
            # 对其他列直接拼接
            merged_data = {column: ''.join(window_df[column].astype(str)) for column in df.columns if column != 'ID'}
            
            # merged_data['ID'] = merged_id
            # merged_dfs.append(merged_data)
            # 创建一个新的行DataFrame，并将其添加到列表中
            merged_df = pd.DataFrame([merged_id], columns=['ID'])
            for key, value in merged_data.items():
                merged_df[key] = [value]
                
            merged_dfs.append(merged_df)

    # 使用concat合并所有的DataFrame
    df_window = pd.concat(merged_dfs, ignore_index=True)
    return df_window

def execute(phe_path, gen_path):
    
    xxx = sliding_window(gen_path)

    y = pd.read_csv(phe_path)
    y = y.iloc[:,1].values
    feature_names = xxx.iloc[:,0].values
    xx = xxx.iloc[:,1:].values
    x = xx.T

    # 拆分训练集和测试集
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    x_train = x
    y_train = y

    # 开始训练
    forest = RandomForestRegressor(random_state=0, n_estimators=200)
    forest.fit(x_train, y_train)

    # 获取特征重要性
    feature_importances = forest.feature_importances_
    
    # 从大到小排序
    sorted_indices = np.argsort(feature_importances)[::-1]
    snp_num = int(1 + len(sorted_indices) * top)
    
    # 输出前10%变量特征名称到csv
    top_feature = feature_names[sorted_indices[:snp_num]]
    df = pd.DataFrame(top_feature)
    df.to_csv('FeatureName.csv', index=False)
    
    genFileHandler(xxx,gen_path,top_feature)

    # 绘制特征的重要性条形图
    # plt.barh(feature_names[sorted_indices[:snp_num]], feature_importances[sorted_indices[:snp_num]])
    # plt.xlabel("Feature Importance")
    # plt.ylabel("Feature Name")
    # plt.title("Feature Importance Plot")
    # plt.savefig('feature_sw.png')
    # plt.close()
    

def genFileHandler(gen,old_gen_path,snp_name):
    # 原地设置索引-取出top10%的snp组合
    gen.set_index(gen.columns[0], inplace=True)
    df = gen.loc[snp_name]
    df.to_csv('newGen_sw.csv', index=True)
    
    # 将独立的位点取出来-并取出整个点位的10%
    xxx = pd.read_csv(old_gen_path, index_col=0)
    snp_num = int(1 + len(xxx.index) * top)
    count = 0
    ordered_set = OrderedDict()
    index_list = df.index.to_list()
    for item in index_list:
        for identity in item.split('#'):
            if count < snp_num:
                ordered_set[identity] = None
                count = len(ordered_set)

    newGen = xxx.loc[list(ordered_set.keys())]
    newGen.to_csv('newGen_sw_.csv', index=True)
    print(newGen)

    

