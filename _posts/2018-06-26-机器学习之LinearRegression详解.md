---
layout: post
title: 机器学习之LinearRegression详解
---

线性回归模型和逻辑回归模型是一种参数机器学习方法。与k-近邻算法不同的是，该方法训练的结果是找到一个最能描述训练集的模式的数学函数。在机器学习里，这样的函数被称为模型。下载本文所需数据集请点击[AmesHousing.txt](/assets/AmesHousing.txt)

## 1. 一般工作流
```python
import pandas as pd
data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]
print(train.info())
target = 'SalePrice'

import matplotlib.pyplot as plt
import seaborn
fig = plt.figure(figsize=(7,15))
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
train.plot(x="Garage Area", y="SalePrice", ax=ax1, kind="scatter")
train.plot(x="Gr Liv Area", y="SalePrice", ax=ax2, kind="scatter")
train.plot(x="Overall Cond", y="SalePrice", ax=ax3, kind="scatter")
plt.show()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train[['Gr Liv Area']], train['SalePrice'])
a0 = lr.intercept_
a1 = lr.coef_

from sklearn.metrics import mean_squared_error
train_predictions = lr.predict(train[['Gr Liv Area']])
test_predictions = lr.predict(test[['Gr Liv Area']])
train_mse = mean_squared_error(train_predictions, train['SalePrice'])
test_mse = mean_squared_error(test_predictions, test['SalePrice'])
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
print(train_rmse)
print(test_rmse)
```

## 2. 特征选择
```python
import pandas as pd
data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]

# 选择数值列，去掉无意义的数值列，和有缺失值的列
numerical_train = train.select_dtypes(include=['int', 'float'])
numerical_train = numerical_train.drop(['PID', 'Year Built', 'Year Remod/Add', 'Garage Yr Blt', 'Mo Sold', 'Yr Sold'], axis=1)
null_series = numerical_train.isnull().sum()
full_cols_series = null_series[null_series == 0]
print(full_cols_series)

# 找出与目标列强相关的列，再应用heatmap找出特征列中互相强相关的列
train_subset = train[full_cols_series.index]
corrmat = train_subset.corr()
sorted_corrs = corrmat['SalePrice'].abs().sort_values()
print(sorted_corrs)
import seaborn as sns
import matplotlib.pyplot as plt
strong_corrs = sorted_corrs[sorted_corrs > 0.3]
corrmat = train_subset[strong_corrs.index].corr()
sns.heatmap(corrmat)

# 利用清洗后的数据训练模型，测试集也要清洗
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
final_corr_cols = strong_corrs.drop(['Garage Cars', 'TotRms AbvGrd'])
features = final_corr_cols.drop(['SalePrice']).index
target = 'SalePrice'
clean_test = test[final_corr_cols.index].dropna()

lr = LinearRegression()
lr.fit(train[features], train['SalePrice'])
train_predictions = lr.predict(train[features])
test_predictions = lr.predict(clean_test[features])

train_mse = mean_squared_error(train_predictions, train[target])
test_mse = mean_squared_error(test_predictions, clean_test[target])
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
print(train_rmse)
print(test_rmse)

# removing features with low variance，也就是数据几乎不变的列
unit_train = (train[features] - train[features].min())/(train[features].max() - train[features].min())
print(unit_train.min())
print(unit_train.max())
# Confirmed: the min and max values are 0.0 and 1.0 respectively
sorted_vars = unit_train.var().sort_values()
print(sorted_vars)
# 可设定阀值剔除低方差的特征列，余同
```

## 3. 梯度下降（Gradient Descent）

以一次线性方程为例（an与a1类似）:
\hat{y} = a_1x_1 + a_0 
MSE = \frac{1}{n} \sum_{i=1}^{n} ({\hat{y_i} - y_i})^2 
MSE(a_0, a_1) = \frac{1}{n} \sum_{i=1}^{n} (a_0 + a_1x_1^{(i)} - y^{(i)} ) ^2 
\frac{d}{da_1} MSE(a_0, a_1) = \frac{2}{n} \sum_{i=1}^{n} x_1^{(i)}(a_0 + a_1x_1^{(i)} - y^{(i)}) 
\frac{d}{da_0} MSE(a_0, a_1) = \frac{2}{n} \sum_{i=1}^{n} (a_0 + a_1x_1^{(i)} - y^{(i)}) 
a_0 := a_0 - \alpha \frac{d}{da_0} MSE(a_0, a_1) 
a_1 := a_1 - \alpha \frac{d}{da_1} MSE(a_0, a_1) 

Gradient Descent是一种Numerical solution，也就是数值解。根据以上公式编写代码：
```python
def a1_derivative(a0, a1, xi_list, yi_list):
    len_data = len(xi_list)
    error = 0
    for i in range(0, len_data):
        error += xi_list[i]*(a0 + a1*xi_list[i] - yi_list[i])
    deriv = 2*error/len_data
    return deriv

def a0_derivative(a0, a1, xi_list, yi_list):
    len_data = len(xi_list)
    error = 0
    for i in range(0, len_data):
        error += a0 + a1*xi_list[i] - yi_list[i]
    deriv = 2*error/len_data
    return deriv

def gradient_descent(xi_list, yi_list, max_iterations, alpha, a1_initial, a0_initial):
    a1_list = [a1_initial]
    a0_list = [a0_initial]

    for i in range(0, max_iterations):
        a1 = a1_list[i]
        a0 = a0_list[i]
        
        a1_deriv = a1_derivative(a0, a1, xi_list, yi_list)
        a0_deriv = a0_derivative(a0, a1, xi_list, yi_list)
        
        a1_new = a1 - alpha*a1_deriv
        a0_new = a0 - alpha*a0_deriv
        
        a1_list.append(a1_new)
        a0_list.append(a0_new)
    return(a0_list, a1_list)

a0_params, a1_params = gradient_descent(train['Gr Liv Area'], train['SalePrice'], 20, .0000003, 150, 1000)
print(a0_params)
print(a1_params)
```

## 4. 普通最小二乘法(ordinary least squares estimation/ OLS)

OLS是一种closed form solution，也就是解析解。其数学原理如下：
\hat{y} = a_0 + a_1x_1 + a_2x_2 + ... + a_nx_n 
Xa = \hat{y} 
\epsilon = \hat{y} - y
y = Xa - \epsilon 
J(a) = \dfrac{1}{n} (Xa - y)^T(Xa - y) 
\frac{dJ(a)}{da} = 2X^TXa - 2X^Ty 
2X^TXa - 2X^Ty = 0 
a = (X^TX)^{-1} X^Ty 
OLS最大的限制是计算复杂度太大，因为矩阵转置的计算复杂度为O(n^3)。所以OLS在数据集小于百万级广泛使用，而在更大级别的数据集中，梯度下降法应用广泛。此外，我们也可以设定阀值，减少计算成本获得近似解。
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]

features = ['Wood Deck SF', 'Fireplaces', 'Full Bath', '1st Flr SF', 'Garage Area', 'Gr Liv Area', 'Overall Qual']
X = train[features]
y = train['SalePrice']

first_term = np.linalg.inv(
        np.dot(
            np.transpose(X), 
            X
        )
    )
second_term = np.dot(
        np.transpose(X),
        y
    )
a = np.dot(first_term, second_term)
print(a)
```

## 5. 总结
线性回归模型LinearRegression进行模型拟合时采用的是ordinary least squares estimation。

---
Enjoy the journey!