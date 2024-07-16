#皮尔森相关系数计算
import math

def pearson_correlation(list1, list2):
    n = len(list1)
    sum_x = sum(list1)
    sum_y = sum(list2)
    sum_xy = sum([list1[i] * list2[i] for i in range(n)])
    sum_x2 = sum([list1[i]**2 for i in range(n)])
    sum_y2 = sum([list2[i]**2 for i in range(n)])
 
    # 计算平均值
    u_x = sum_x / n
    u_y = sum_y / n
 
    # 计算标准差
    std_dev_x = math.sqrt(sum_x2 / n - u_x**2)
    std_dev_y = math.sqrt(sum_y2 / n - u_y**2)
 
    # 计算皮尔森相关系数
    if std_dev_x > 0 and std_dev_y > 0:
        pearson = (sum_xy / n - u_x * u_y) / (std_dev_x * std_dev_y)
    else:
        pearson = 0
 
    return pearson
 
# 示例使用
list1 = [1, 2, 3, 4, 5]
list2 = [5, 4, 3, 2, 1]
print(pearson_correlation(list1, list2))  # 输出相关系数