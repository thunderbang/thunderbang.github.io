---
layout: post
title: Feature Extraction Process of JBRJ
---

**Notes**: 
1. 点击超链接可直达正文；
2. 以下第二节中粗体为模型使用过的特征。

### 0、[Introduction](#0)
<!--more-->
### 1、[Import related libraries & Define related function](#1)

### 2、[Feature extraction from jbrj database](#2)

2.1、[t_voicecall](#2.1)
- dial_ratio：通话记录中主叫次数/总通话次数
- **dial_maxmin**：月通话主叫率极差
- **dial_std**：月通话主叫率标准差
- duration_10_ratio：通话时长超过10分钟占比
- duration_5_ratio：通话时长超过5分钟占比

2.2、[t_mobilebill](#2.2)
- mean_fee：平均月话费
- **max_fee**：最大月话费
- min_fee：最小月话费

2.3、[t_mobilebasic](#2.3)
- **open_year**：开卡年数
- level：运营商用户星级
- available_balance：当前话费余额
- level_per_year：星级/年数
    - **level_per_year_log**：星级/年数(对数处理)

2.4、[loan_contacts_detail_list](#2.4)
- **city_number**：通话记录中通话城市个数

2.5、[user](#2.5)
- **device_type**：手机型号(1安卓，2苹果)

2.6、[loan_contacts_detail](#2.6)
- contact_num：常用联系人数量
- frequent_contact_num：频繁联系人数量
- **contact_1_times**：第一联系人次数
- contact_2_times：第二联系人次数

2.7、[t_reportdata](#2.7)
- **poweroff_days**：180天内关机天数
- **poweroff_times**：连续3天以上关机次数

2.8、[tongdun_report](#2.8)
- **age**：年龄
- tongdun_score：同盾得分
- loan_7_num：7天借贷平台数
- loan_30_num：30天借贷平台数
    - **loan_30_num_log**：30天借贷平台数(对数处理)

2.9、[xinyan_wash_black & xinyan_wash_white](#2.9)
- **xinyan_score**：新颜得分(0洗白，1拉黑)

2.10、[magic_wand_v2](#2.10)
- black_score：涉黑得分, 越大越好, 越小涉黑越深
- **auth_contactnum180**：180天授权过的直接联系人数
- auth_contactnum90：90天授权过的直接联系人数
- auth_contactnum30：30天授权过的直接联系人数
- mobile_other_devices_cnt：手机号关联的设备数
- **idcard_other_devices_cnt**：身份证关联的设备数
- other_devices_cnt：
- overdue_count：逾期次数
- **is_overdue**：是否逾期

### 3、[Compute corrcoef & IV value](#3)

### 4、[Reflection and summary](#4)


## <span id='0'>0、Introduction</span>

本产品反欺诈包含4个模型，每个模型包括GradientBoostingClassifier (GBDT)、XGBClassifier (XGboost)和RandomForestClassifier (RF)三种集成算法，结果由三个算法投票确定。不同模型特征略有差异，算法的阈值也不尽相同。


```python
V2.1.1:
    ['open_year', 'device_type', 'max_fee', 'contact1_num', 'city_num', 
     'xinyan_score', 'level_per_year_2_new', 'loan_30_num_2_new', 'poweroff_days',
     'poweroff_times', 'dial_std', 'dial_maxmin', 'age', 'auth_contactnum180',
     'overdue_yesno', 'idcard_other_devices_cnt']
    
V2.1.2:
    ['open_year', 'device_type', 'max_fee', 'contact1_num', 'city_num',
     'xinyan_score', 'loan_30_num_2_new', 'poweroff_days', 'poweroff_times',
     'dial_std', 'dial_maxmin', 'age', 'auth_contactnum180', 'overdue_yesno', 
     'idcard_other_devices_cnt']
    
V2.1.3:     
    ['open_year', 'device_type', 'max_fee', 'contact1_num', 'city_num',
     'level_per_year_2_new', 'loan_30_num_2_new', 'poweroff_days', 'poweroff_times',
     'dial_std', 'dial_maxmin', 'age', 'auth_contactnum180', 'overdue_yesno', 
     'idcard_other_devices_cnt']
     
V2.1.4:
    ['open_year', 'device_type', 'max_fee', 'contact1_num', 'city_num',
     'loan_30_num_2_new', 'poweroff_days', 'poweroff_times', 'dial_std', 'dial_maxmin', 
     'age', 'auth_contactnum180', 'overdue_yesno', 'idcard_other_devices_cnt']    
     
notes:
    修改特征命名：overdue_yesno 改为 is_overdue
                 level_per_year_2_new 改为 level_per_year_log
                 loan_30_num_2_new 改为 loan_30_num_log
                 contact1_num 改为 contact_1_times
```

## <span id='1'>1、Import related libraries & Define related function</span>


```python
import numpy as np
import pandas as pd
import pymysql
from scipy import stats
import math
```


```python
import re

def parse_json(x, regex):
    values = regex.findall(x)
    if values:
        return int(values[0])
    else:
        return np.nan
```


```python
def iv(dataframe,tag,cut = False, n = 10):
    #bad value equals to 1
    #good value equals to 0
    #cut = FASLE : discrete data ,no bin division
    #cut = 1 : optimal bin division
    bad = dataframe['mark'].sum()
    good = len(dataframe['mark']) - bad
    r = 0
    if cut == 1:
        while np.abs(r) < 1: 
            n = n - 1
            dataframe['qcut'] = pd.qcut(dataframe[tag],n)
            d = dataframe.groupby('qcut')
            #d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)}) 
            #d2 = d1.groupby('Bucket', as_index = True) 
            r, p = stats.spearmanr(d.mean()[tag], d.mean().mark) 
    elif cut == 2:
        dataframe['qcut'] = pd.qcut(dataframe[tag],n)
        d = dataframe.groupby('qcut')
    else:
        d = dataframe.groupby([tag])
    d1 = pd.concat([d.sum()[['mark']],d.count()[['mark']]],axis=1)
    d1.columns=['bi','ni']
    #print(d1)
    d1['woe'] = np.log(((d1['ni']-d1['bi'])/good)/(d1['bi']/bad))
    print('woe:\n',d1['woe'])
    d1['iv'] = ((d1['ni']-d1['bi'])/good - d1['bi']/bad)*d1['woe']
    print('iv:\n',d1['iv'])
    if cut:
        return (d1['iv'].sum(),n)
    else:
        return d1['iv'].sum()   #cut = 2 : qcut 
  
```


```python
conn = pymysql.connect(host='xxxxxx',user='xxx',password='xxx',db='xxx',charset='utf8')
```

## <span id='2'>2、Feature extraction from jbrj database</span>

### <span id='2.1'>2.1、t_voicecall</span>
- dial_ratio：通话记录中主叫次数/总通话次数
- **dial_maxmin**：月通话主叫率极差
- **dial_std**：月通话主叫率标准差
- duration_10_ratio：通话时长超过10分钟占比
- duration_5_ratio：通话时长超过5分钟占比


```python
sql_voicecall = '''
SELECT 
    SUBSTR(tv.userId, 7) user_id,
    tv.dialType,
    tv.billMonth,
    tv.durationInSecond duration 
FROM %s tv
INNER JOIN table4_pyy_2 t
ON t.user_id = SUBSTR(tv.userId, 7)
'''%('t_voicecall_00')

voice_call_df = pd.read_sql(sql_voicecall, conn)

for i in range(1, 100):
    table_name = 't_voicecall_' + str(i).zfill(2)
    sql_voicecall = '''
        SELECT 
            SUBSTR(tv.userId, 7) user_id,
            tv.dialType,
            tv.billMonth,
            tv.durationInSecond duration 
        FROM %s tv
        INNER JOIN table4_pyy_2 t
        ON t.user_id = SUBSTR(tv.userId, 7)
        '''%(table_name)
    df = pd.read_sql(sql_voicecall, conn)
    voice_call_df = voice_call_df.append(df)

print(voice_call_df.shape)
voice_call_df.head()
```

    (3414461, 4)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>dialType</th>
      <th>billMonth</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48900</td>
      <td>DIALED</td>
      <td>2018-06</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48900</td>
      <td>DIAL</td>
      <td>2018-06</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48900</td>
      <td>DIALED</td>
      <td>2018-06</td>
      <td>22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48900</td>
      <td>DIALED</td>
      <td>2018-06</td>
      <td>18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48900</td>
      <td>DIALED</td>
      <td>2018-06</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>




```python
voice_call_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3414461 entries, 0 to 18530
    Data columns (total 4 columns):
    user_id      object
    dialType     object
    billMonth    object
    duration     int64
    dtypes: int64(1), object(3)
    memory usage: 130.3+ MB
    


```python
voice_call_df['dialType'].unique()
```




    array(['DIALED', 'DIAL', 'NULL'], dtype=object)




```python
voice_call_df['dialType'].replace({'DIAL':1, 'DIALED':0, 'NULL': np.nan}, inplace=True)
voice_call_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3414461 entries, 0 to 18530
    Data columns (total 4 columns):
    user_id      object
    dialType     float64
    billMonth    object
    duration     int64
    dtypes: float64(1), int64(1), object(2)
    memory usage: 130.3+ MB
    


```python
voice_call_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>dialType</th>
      <th>billMonth</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48900</td>
      <td>0.0</td>
      <td>2018-06</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48900</td>
      <td>1.0</td>
      <td>2018-06</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48900</td>
      <td>0.0</td>
      <td>2018-06</td>
      <td>22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48900</td>
      <td>0.0</td>
      <td>2018-06</td>
      <td>18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48900</td>
      <td>0.0</td>
      <td>2018-06</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>




```python
dial_ratio_df = voice_call_df.groupby('user_id').mean()['dialType'].to_frame('dial_ratio')
dial_ratio_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dial_ratio</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15497</th>
      <td>0.307573</td>
    </tr>
    <tr>
      <th>15531</th>
      <td>0.388417</td>
    </tr>
    <tr>
      <th>27232</th>
      <td>0.473896</td>
    </tr>
    <tr>
      <th>28414</th>
      <td>0.371875</td>
    </tr>
    <tr>
      <th>29428</th>
      <td>0.366310</td>
    </tr>
  </tbody>
</table>
</div>




```python
dial_ratio_df.shape
```




    (1689, 1)




```python
dial_ratio_month = voice_call_df.groupby(['user_id','billMonth'], as_index=False)['dialType'].mean()
dial_ratio_month.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>billMonth</th>
      <th>dialType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15497</td>
      <td>2018-02</td>
      <td>0.449275</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15497</td>
      <td>2018-03</td>
      <td>0.337838</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15497</td>
      <td>2018-04</td>
      <td>0.368715</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15497</td>
      <td>2018-05</td>
      <td>0.260870</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15497</td>
      <td>2018-06</td>
      <td>0.234483</td>
    </tr>
  </tbody>
</table>
</div>




```python
dial_maxmin = dial_ratio_month.groupby('user_id')['dialType'].max()-dial_ratio_month.groupby('user_id')['dialType'].min()
dial_maxmin_df = dial_maxmin.to_frame('dial_maxmin')
dial_maxmin_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dial_maxmin</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15497</th>
      <td>0.396644</td>
    </tr>
    <tr>
      <th>15531</th>
      <td>0.335450</td>
    </tr>
    <tr>
      <th>27232</th>
      <td>0.261487</td>
    </tr>
    <tr>
      <th>28414</th>
      <td>0.153243</td>
    </tr>
    <tr>
      <th>29428</th>
      <td>0.368590</td>
    </tr>
  </tbody>
</table>
</div>




```python
dial_std = dial_ratio_month.groupby('user_id')['dialType'].std()
dial_std_df = dial_std.to_frame('dial_std')
dial_std_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dial_std</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15497</th>
      <td>0.137047</td>
    </tr>
    <tr>
      <th>15531</th>
      <td>0.147918</td>
    </tr>
    <tr>
      <th>27232</th>
      <td>0.095122</td>
    </tr>
    <tr>
      <th>28414</th>
      <td>0.057422</td>
    </tr>
    <tr>
      <th>29428</th>
      <td>0.132449</td>
    </tr>
  </tbody>
</table>
</div>




```python
duration_count = voice_call_df.groupby('user_id').count()['duration'].to_frame('count')
duration_count_10 = voice_call_df[voice_call_df['duration'] >= 60*10].groupby('user_id').count()['duration'].to_frame('count_10')
duration_count_5 = voice_call_df[voice_call_df['duration'] >= 60*5].groupby('user_id').count()['duration'].to_frame('count_5')
```


```python
duration_df = pd.merge(duration_count, duration_count_10, how='left', left_index=True, right_index=True)
duration_df = pd.merge(duration_df, duration_count_5, how='left', left_index=True, right_index=True)
```


```python
duration_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>count_10</th>
      <th>count_5</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15497</th>
      <td>647</td>
      <td>16.0</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>15531</th>
      <td>1295</td>
      <td>78.0</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>27232</th>
      <td>1245</td>
      <td>12.0</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>28414</th>
      <td>1280</td>
      <td>12.0</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>29428</th>
      <td>1122</td>
      <td>16.0</td>
      <td>49.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
duration_df['duration_10_ratio'] = duration_df['count_10'] / duration_df['count']
duration_df['duration_5_ratio'] = duration_df['count_5'] / duration_df['count']
```


```python
duration_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>count_10</th>
      <th>count_5</th>
      <th>duration_10_ratio</th>
      <th>duration_5_ratio</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15497</th>
      <td>647</td>
      <td>16.0</td>
      <td>47.0</td>
      <td>0.024730</td>
      <td>0.072643</td>
    </tr>
    <tr>
      <th>15531</th>
      <td>1295</td>
      <td>78.0</td>
      <td>120.0</td>
      <td>0.060232</td>
      <td>0.092664</td>
    </tr>
    <tr>
      <th>27232</th>
      <td>1245</td>
      <td>12.0</td>
      <td>54.0</td>
      <td>0.009639</td>
      <td>0.043373</td>
    </tr>
    <tr>
      <th>28414</th>
      <td>1280</td>
      <td>12.0</td>
      <td>38.0</td>
      <td>0.009375</td>
      <td>0.029687</td>
    </tr>
    <tr>
      <th>29428</th>
      <td>1122</td>
      <td>16.0</td>
      <td>49.0</td>
      <td>0.014260</td>
      <td>0.043672</td>
    </tr>
  </tbody>
</table>
</div>



### <span id='2.2'>2.2、t_mobilebill</span>
- mean_fee：平均月话费
- **max_fee**：最大月话费
- min_fee：最小月话费


```python
sql_mobilebill = '''
SELECT 
    SUBSTR(tm.userId, 7) user_id,
    tm.billMonth,
    tm.totalFee fee
FROM t_mobilebill tm
INNER JOIN table4_pyy_2 t
ON t.user_id = SUBSTR(tm.userId, 7)
'''

mobile_bill_df = pd.read_sql(sql_mobilebill, conn)
print(mobile_bill_df.shape)
mobile_bill_df.head()
```

    (10059, 3)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>billMonth</th>
      <th>fee</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32977</td>
      <td>2018-02</td>
      <td>2850</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32977</td>
      <td>2018-03</td>
      <td>4605</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32977</td>
      <td>2017-12</td>
      <td>6931</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32977</td>
      <td>2018-04</td>
      <td>12724</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32977</td>
      <td>2018-05</td>
      <td>6639</td>
    </tr>
  </tbody>
</table>
</div>




```python
mobile_bill_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10059 entries, 0 to 10058
    Data columns (total 3 columns):
    user_id      10059 non-null object
    billMonth    10059 non-null object
    fee          10059 non-null int64
    dtypes: int64(1), object(2)
    memory usage: 235.8+ KB
    


```python
max_fee = mobile_bill_df.groupby('user_id')['fee'].max().to_frame('max_fee')
min_fee = mobile_bill_df.groupby('user_id')['fee'].min().to_frame('min_fee')
mean_fee = mobile_bill_df.groupby('user_id')['fee'].mean().to_frame('mean_fee')
```


```python
mobile_bill_df = pd.merge(max_fee, min_fee, how='left', left_index=True, right_index=True)
mobile_bill_df = pd.merge(mobile_bill_df, mean_fee, how='left', left_index=True, right_index=True)
print(mobile_bill_df.shape)
mobile_bill_df.head()
```

    (1691, 3)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_fee</th>
      <th>min_fee</th>
      <th>mean_fee</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15497</th>
      <td>41700</td>
      <td>17700</td>
      <td>22150.000000</td>
    </tr>
    <tr>
      <th>15531</th>
      <td>8682</td>
      <td>0</td>
      <td>5406.666667</td>
    </tr>
    <tr>
      <th>27232</th>
      <td>6433</td>
      <td>900</td>
      <td>3696.500000</td>
    </tr>
    <tr>
      <th>28414</th>
      <td>49033</td>
      <td>11800</td>
      <td>18005.500000</td>
    </tr>
    <tr>
      <th>29428</th>
      <td>39800</td>
      <td>19900</td>
      <td>29853.333333</td>
    </tr>
  </tbody>
</table>
</div>



### <span id='2.3'>2.3、t_mobilebasic</span>
- **open_year**：开卡年数
- level：运营商用户星级
- available_balance：当前话费余额
- level_per_year：星级/年数
    - **level_per_year_log**：星级/年数(对数处理)


```python
sql_mobilebasic = '''
SELECT 
    SUBSTR(tm.userId, 7) user_id,
    DATEDIFF('2018-08-21',tm.openTime)/365 open_year,
    (CASE
        WHEN tm.level LIKE '%一%' THEN 1
        WHEN tm.level LIKE '%二%' THEN 2
        WHEN tm.level LIKE '%三%' THEN 3
        WHEN tm.level LIKE '%四%' THEN 4
        WHEN tm.level LIKE '%五%' THEN 5
        WHEN tm.level LIKE '%普通%' THEN 0
    END) level,
    tm.availableBalance
FROM t_mobilebasic tm
INNER JOIN table4_pyy_2 t
ON t.user_id = SUBSTR(tm.userId, 7)
'''
mobile_basic_df = pd.read_sql(sql_mobilebasic, conn)
print(mobile_basic_df.shape)
mobile_basic_df.head()
```

    (1691, 4)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>open_year</th>
      <th>level</th>
      <th>availableBalance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32977</td>
      <td>1.4493</td>
      <td>3.0</td>
      <td>245.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33118</td>
      <td>1.4849</td>
      <td>2.0</td>
      <td>1907.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33369</td>
      <td>1.0027</td>
      <td>NaN</td>
      <td>1033.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33531</td>
      <td>2.8849</td>
      <td>3.0</td>
      <td>-128.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33750</td>
      <td>1.0301</td>
      <td>NaN</td>
      <td>1701.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
mobile_basic_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1691 entries, 0 to 1690
    Data columns (total 4 columns):
    user_id             1691 non-null object
    open_year           1572 non-null float64
    level               1526 non-null float64
    availableBalance    1676 non-null float64
    dtypes: float64(3), object(1)
    memory usage: 52.9+ KB
    


```python
mobile_basic_df['level_per_year'] = mobile_basic_df['level'] / mobile_basic_df['open_year']
mobile_basic_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>open_year</th>
      <th>level</th>
      <th>availableBalance</th>
      <th>level_per_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32977</td>
      <td>1.4493</td>
      <td>3.0</td>
      <td>245.0</td>
      <td>2.069965</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33118</td>
      <td>1.4849</td>
      <td>2.0</td>
      <td>1907.0</td>
      <td>1.346892</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33369</td>
      <td>1.0027</td>
      <td>NaN</td>
      <td>1033.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33531</td>
      <td>2.8849</td>
      <td>3.0</td>
      <td>-128.0</td>
      <td>1.039897</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33750</td>
      <td>1.0301</td>
      <td>NaN</td>
      <td>1701.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# mobile_basic_df['level_per_year_log']
```

### <span id='2.4'>2.4、loan_contacts_detail_list</span>
- **city_number**：通话记录中通话城市个数


```python
sql_lcdl = '''
SELECT 
    t.user_id,
    a.city_num
FROM (
    SELECT 
        tr.user_id,
        count(DISTINCT lcdl.city) city_num
    FROM loan_contacts_detail_list lcdl 
    INNER JOIN tongdun_report tr
    ON lcdl.phone_num = tr.mobile
    GROUP BY lcdl.phone_num
    ) a
INNER JOIN table4_pyy_2 t
ON t.user_id = a.user_id
'''
city_number_df = pd.read_sql(sql_lcdl, conn)
print(city_number_df.shape)
city_number_df.head()
```

    (1689, 2)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>city_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15497</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27232</td>
      <td>42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33118</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35223</td>
      <td>40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36596</td>
      <td>52</td>
    </tr>
  </tbody>
</table>
</div>




```python
city_number_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1689 entries, 0 to 1688
    Data columns (total 2 columns):
    user_id     1689 non-null int64
    city_num    1689 non-null int64
    dtypes: int64(2)
    memory usage: 26.5 KB
    


```python
city_number_df['user_id'] = city_number_df['user_id'].astype('str')
```

### <span id='2.5'>2.5、user</span>
- **device_type**：手机型号(1安卓，2苹果)


```python
sql_user = '''
SELECT 
    u.id user_id,
    u.platform device_type,
    t.mark
FROM user u
INNER JOIN table4_pyy_2 t
ON t.user_id = u.id
'''
user_device_df = pd.read_sql(sql_user, conn)
print(user_device_df.shape)
user_device_df.head()
```

    (1691, 3)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>device_type</th>
      <th>mark</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15497</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27232</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33118</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35223</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36596</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_device_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1691 entries, 0 to 1690
    Data columns (total 3 columns):
    user_id        1691 non-null int64
    device_type    1691 non-null object
    mark           1691 non-null object
    dtypes: int64(1), object(2)
    memory usage: 39.7+ KB
    


```python
user_device_df['user_id'] = user_device_df['user_id'].astype('str')
```

### <span id='2.6'>2.6、loan_contacts_detail</span>
- contact_num：常用联系人数量
- frequent_contact_num：频繁联系人数量
- **contact_1_times**：第一联系人次数
- contact_2_times：第二联系人次数


```python
sql_contact = '''
SELECT 
    l.user_id,
    count(*) contact_num 
FROM loan_contacts_detail l
INNER JOIN table4_pyy_2 t
ON t.user_id = l.user_id
WHERE l.dict_num REGEXP'1[3|5|8|4|7][0-9]{9}'
GROUP BY 1
'''
contact_num = pd.read_sql(sql_contact, conn)
print(contact_num.shape)
contact_num.head()
```

    (1687, 2)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>contact_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15497</td>
      <td>128</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15531</td>
      <td>197</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27232</td>
      <td>66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28414</td>
      <td>42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29428</td>
      <td>66</td>
    </tr>
  </tbody>
</table>
</div>




```python
sql_frequent_contact = '''
SELECT 
    l.user_id,
    count(*) frequent_contact_num 
FROM loan_contacts_detail l
INNER JOIN table4_pyy_2 t
ON t.user_id = l.user_id
WHERE l.dict_num REGEXP'1[3|5|8|4|7][0-9]{9}'
AND l.dict_type LIKE '%频繁%'
GROUP BY 1
'''
frequent_contact_num = pd.read_sql(sql_frequent_contact, conn)
print(frequent_contact_num.shape)
frequent_contact_num.head()
```

    (1675, 2)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>frequent_contact_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15497</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15531</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27232</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28414</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29428</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
sql_contact_1 = '''
SELECT 
    l.user_id,
    l.contact_time contact_1_times 
FROM loan_contacts_detail l
INNER JOIN table4_pyy_2 t
ON t.user_id = l.user_id
WHERE l.dict_type LIKE '%第一%'
'''
contact_1_times = pd.read_sql(sql_contact_1, conn)
print(contact_1_times.shape)
contact_1_times.head()
```

    (1881, 2)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>contact_1_times</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15497</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27232</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35223</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36596</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37799</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
contact_1_times.drop_duplicates('user_id', inplace=True)
contact_1_times.shape
```




    (1675, 2)




```python
sql_contact_2 = '''
SELECT 
    l.user_id,
    l.contact_time contact_2_times 
FROM loan_contacts_detail l
INNER JOIN table4_pyy_2 t
ON t.user_id = l.user_id
WHERE l.dict_type LIKE '%第二%'
'''
contact_2_times = pd.read_sql(sql_contact_2, conn)
print(contact_2_times.shape)
contact_2_times.head()
```

    (1794, 2)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>contact_2_times</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15497</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27232</td>
      <td>63</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35223</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36596</td>
      <td>237</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37993</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
contact_2_times.drop_duplicates('user_id', inplace=True)
contact_2_times.shape
```




    (1676, 2)




```python
contact_num.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1687 entries, 0 to 1686
    Data columns (total 2 columns):
    user_id        1687 non-null int64
    contact_num    1687 non-null int64
    dtypes: int64(2)
    memory usage: 26.4 KB
    


```python
contact_num['user_id'] = contact_num['user_id'].astype('str')
frequent_contact_num['user_id'] = frequent_contact_num['user_id'].astype('str')
contact_1_times['user_id'] = contact_1_times['user_id'].astype('str')
contact_2_times['user_id'] = contact_2_times['user_id'].astype('str')
```

### <span id='2.7'>2.7、t_reportdata</span>
- **poweroff_days**：180天内关机天数
- **poweroff_times**：连续3天以上关机次数


```python
sql_t_reportdata = '''
SELECT 
    t.user_id,
    tr.reportData
FROM t_reportdata tr
INNER JOIN table4_pyy_2 t
ON t.user_id = SUBSTR(tr.userId, 7)
'''
poweroff_df = pd.read_sql(sql_t_reportdata, conn)
print(poweroff_df.shape)
poweroff_df.head()
```

    (1691, 2)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>reportData</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32977</td>
      <td>{"report":[{"key":"data_type","value":"运营商"},{...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33118</td>
      <td>{"report":[{"key":"data_type","value":"运营商"},{...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33369</td>
      <td>{"report":[{"key":"data_type","value":"运营商"},{...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33531</td>
      <td>{"report":[{"key":"data_type","value":"运营商"},{...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33750</td>
      <td>{"report":[{"key":"data_type","value":"运营商"},{...</td>
    </tr>
  </tbody>
</table>
</div>




```python
regex_poweroff_days = re.compile('根据运营商详单数据，180天内关机(\d+)天')

poweroff_df['poweroff_days'] = poweroff_df['reportData'].apply(parse_json, args=(regex_poweroff_days,))
```


```python
regex_poweroff_times = re.compile('连续三天以上关机(\d+)次')

poweroff_df['poweroff_times'] = poweroff_df['reportData'].apply(parse_json, args=(regex_poweroff_times,))
```


```python
poweroff_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>reportData</th>
      <th>poweroff_days</th>
      <th>poweroff_times</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32977</td>
      <td>{"report":[{"key":"data_type","value":"运营商"},{...</td>
      <td>23.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33118</td>
      <td>{"report":[{"key":"data_type","value":"运营商"},{...</td>
      <td>47.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33369</td>
      <td>{"report":[{"key":"data_type","value":"运营商"},{...</td>
      <td>6.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33531</td>
      <td>{"report":[{"key":"data_type","value":"运营商"},{...</td>
      <td>25.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33750</td>
      <td>{"report":[{"key":"data_type","value":"运营商"},{...</td>
      <td>44.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
poweroff_df.drop('reportData', axis=1, inplace=True)
```


```python
poweroff_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1691 entries, 0 to 1690
    Data columns (total 3 columns):
    user_id           1691 non-null int64
    poweroff_days     1326 non-null float64
    poweroff_times    1326 non-null float64
    dtypes: float64(2), int64(1)
    memory usage: 39.7 KB
    


```python
poweroff_df['user_id'] = poweroff_df['user_id'].astype('str')
```

### <span id='2.8'>2.8、tongdun_report</span>
- **age**：年龄
- tongdun_score：同盾得分
- loan_7_num：7天借贷平台数
- loan_30_num：30天借贷平台数
    - **loan_30_num_log**：30天借贷平台数(对数处理)


```python
sql_tongdun_report = '''
SELECT 
    t.user_id,
    FLOOR(DATEDIFF(DATE(SYSDATE()), DATE(SUBSTR(tr.identity_number,7,8)))/ 365.25) age,
    tr.final_score tongdun_score,
    tr.report_data
FROM tongdun_report tr
INNER JOIN table4_pyy_2 t
ON t.user_id = tr.user_id
'''
tongdun_df = pd.read_sql(sql_tongdun_report, conn)
print(tongdun_df.shape)
tongdun_df.head()
```

    (1692, 4)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>age</th>
      <th>tongdun_score</th>
      <th>report_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15497</td>
      <td>24</td>
      <td>91</td>
      <td>{"success":true,"result_desc":{"INFOANALYSIS":...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27232</td>
      <td>30</td>
      <td>49</td>
      <td>{"success":true,"result_desc":{"INFOANALYSIS":...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33118</td>
      <td>18</td>
      <td>82</td>
      <td>{"success":true,"result_desc":{"INFOANALYSIS":...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35223</td>
      <td>22</td>
      <td>60</td>
      <td>{"success":true,"result_desc":{"INFOANALYSIS":...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36596</td>
      <td>30</td>
      <td>67</td>
      <td>{"success":true,"result_desc":{"INFOANALYSIS":...</td>
    </tr>
  </tbody>
</table>
</div>




```python
tongdun_df.drop_duplicates('user_id', inplace=True)
tongdun_df.shape
```




    (1691, 4)




```python
loan_7_num_regex = re.compile('7天内申请人在多个平台申请借款.*?platform_count.*?(\d+)', re.DOTALL)

tongdun_df['loan_7_num'] = tongdun_df['report_data'].apply(parse_json, args=(loan_7_num_regex,))
```


```python
loan_30_num_regex = re.compile('1个月内申请人在多个平台申请借款.*?platform_count.*?(\d+)', re.DOTALL)

tongdun_df['loan_30_num'] = tongdun_df['report_data'].apply(parse_json, args=(loan_30_num_regex,))
```


```python
tongdun_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>age</th>
      <th>tongdun_score</th>
      <th>report_data</th>
      <th>loan_7_num</th>
      <th>loan_30_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15497</td>
      <td>24</td>
      <td>91</td>
      <td>{"success":true,"result_desc":{"INFOANALYSIS":...</td>
      <td>17.0</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27232</td>
      <td>30</td>
      <td>49</td>
      <td>{"success":true,"result_desc":{"INFOANALYSIS":...</td>
      <td>3.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33118</td>
      <td>18</td>
      <td>82</td>
      <td>{"success":true,"result_desc":{"INFOANALYSIS":...</td>
      <td>8.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35223</td>
      <td>22</td>
      <td>60</td>
      <td>{"success":true,"result_desc":{"INFOANALYSIS":...</td>
      <td>9.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36596</td>
      <td>30</td>
      <td>67</td>
      <td>{"success":true,"result_desc":{"INFOANALYSIS":...</td>
      <td>9.0</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
tongdun_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1691 entries, 0 to 1691
    Data columns (total 6 columns):
    user_id          1691 non-null int64
    age              1691 non-null int64
    tongdun_score    1691 non-null int64
    report_data      1691 non-null object
    loan_7_num       1630 non-null float64
    loan_30_num      1676 non-null float64
    dtypes: float64(2), int64(3), object(1)
    memory usage: 92.5+ KB
    


```python
tongdun_df.drop('report_data', axis=1, inplace=True)
```


```python
tongdun_df['user_id'] = tongdun_df['user_id'].astype('str')
```

### <span id='2.9'>2.9、xinyan_wash_black & xinyan_wash_white</span>
- **xinyan_score**：新颜得分(0洗白，1拉黑)


```python
sql_xwb = '''
SELECT t.user_id, xwb.code xinyan_score
FROM (
SELECT user_id,code 
FROM xinyan_wash_black
WHERE `code`=0
) xwb
INNER JOIN table4_pyy_2 t
ON t.user_id = xwb.user_id
'''

sql_xww = '''
SELECT t.user_id, xww.code xinyan_score
FROM (
SELECT user_id,code 
FROM xinyan_wash_white
WHERE (code=0 OR code=1)
) xww
INNER JOIN table4_pyy_2 t
ON t.user_id = xww.user_id
'''

xinyan_wash_black = pd.read_sql(sql_xwb, conn)
xinyan_wash_white = pd.read_sql(sql_xww, conn)
```


```python
xinyan_wash_black['xinyan_score'] = 1
xinyan_wash_white['xinyan_score'] = 0
xinyan_df = pd.concat([xinyan_wash_black, xinyan_wash_white], axis=0)
print(xinyan_df.shape)
xinyan_df.head()
```

    (514, 2)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>xinyan_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41061</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47583</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48433</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48537</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48777</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
xinyan_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 514 entries, 0 to 385
    Data columns (total 2 columns):
    user_id         514 non-null int64
    xinyan_score    514 non-null int64
    dtypes: int64(2)
    memory usage: 12.0 KB
    


```python
xinyan_df['user_id'] = xinyan_df['user_id'].astype('str')
```

### <span id='2.10'>2.10、magic_wand_v2</span>
- black_score：涉黑得分, 越大越好, 越小涉黑越深
- **auth_contactnum180**：180天授权过的直接联系人数
- auth_contactnum90：90天授权过的直接联系人数
- auth_contactnum30：30天授权过的直接联系人数
- mobile_other_devices_cnt：手机号关联的设备数
- **idcard_other_devices_cnt**：身份证关联的设备数
- other_devices_cnt：
- overdue_count：逾期次数
- **is_overdue**：是否逾期


```python
sql_magic_wand = '''
SELECT 
    t.user_id,
    m.black_score,
    m.result_data
FROM magic_wand_v2 m 
INNER JOIN table4_pyy_2 t
ON t.user_id=m.user_id
'''

magic_wand = pd.read_sql(sql_magic_wand, conn)
print(magic_wand.shape)
magic_wand.head()
```

    (1307, 3)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>black_score</th>
      <th>result_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15531</td>
      <td>0.0</td>
      <td>{"msg":"操作成功","code":"0000","data":{"gray_info...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15497</td>
      <td>91.0</td>
      <td>{"msg":"操作成功","code":"0000","data":{"gray_info...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27232</td>
      <td>63.0</td>
      <td>{"msg":"操作成功","code":"0000","data":{"gray_info...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28414</td>
      <td>100.0</td>
      <td>{"msg":"操作成功","code":"0000","data":{"gray_info...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29428</td>
      <td>0.0</td>
      <td>{"msg":"操作成功","code":"0000","data":{"gray_info...</td>
    </tr>
  </tbody>
</table>
</div>




```python
auth_contactnum180_regex = re.compile('mobile_contact_180d.*?auth_contactnum.*?(\d+)', re.DOTALL)
auth_contactnum90_regex = re.compile('mobile_contact_90d.*?auth_contactnum.*?(\d+)', re.DOTALL)
auth_contactnum30_regex = re.compile('mobile_contact_30d.*?auth_contactnum.*?(\d+)', re.DOTALL)

magic_wand['auth_contactnum180'] = magic_wand['result_data'].apply(parse_json, args=(auth_contactnum180_regex,))
magic_wand['auth_contactnum90'] = magic_wand['result_data'].apply(parse_json, args=(auth_contactnum90_regex,))
magic_wand['auth_contactnum30'] = magic_wand['result_data'].apply(parse_json, args=(auth_contactnum30_regex,))
```


```python
mobile_other_devices_cnt_regex = re.compile('mobile_other_devices_cnt.*?(\d+)', re.DOTALL)
idcard_other_devices_cnt_regex = re.compile('idcard_other_devices_cnt.*?(\d+)', re.DOTALL)
other_devices_cnt_regex = re.compile('other_devices_cnt.*?(\d+)', re.DOTALL)

magic_wand['mobile_other_devices_cnt'] = magic_wand['result_data'].apply(parse_json, args=(mobile_other_devices_cnt_regex,))
magic_wand['idcard_other_devices_cnt'] = magic_wand['result_data'].apply(parse_json, args=(idcard_other_devices_cnt_regex,))
magic_wand['other_devices_cnt'] = magic_wand['result_data'].apply(parse_json, args=(other_devices_cnt_regex,))
```


```python
overdue_count_regex = re.compile('graylist_record.*?overdue_count.*?(\d+)', re.DOTALL)

magic_wand['overdue_count'] = magic_wand['result_data'].apply(parse_json, args=(overdue_count_regex,))
```


```python
magic_wand['is_overdue'] = magic_wand['overdue_count'].apply(lambda x: 1 if x>0 else 0)
```


```python
magic_wand.drop('result_data', axis=1, inplace=True)
magic_wand.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>black_score</th>
      <th>auth_contactnum180</th>
      <th>auth_contactnum90</th>
      <th>auth_contactnum30</th>
      <th>mobile_other_devices_cnt</th>
      <th>idcard_other_devices_cnt</th>
      <th>other_devices_cnt</th>
      <th>overdue_count</th>
      <th>is_overdue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15531</td>
      <td>0.0</td>
      <td>9</td>
      <td>8</td>
      <td>5</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15497</td>
      <td>91.0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27232</td>
      <td>63.0</td>
      <td>9</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28414</td>
      <td>100.0</td>
      <td>6</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29428</td>
      <td>0.0</td>
      <td>17</td>
      <td>13</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
magic_wand.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1307 entries, 0 to 1306
    Data columns (total 10 columns):
    user_id                     1307 non-null int64
    black_score                 1307 non-null float64
    auth_contactnum180          1307 non-null int64
    auth_contactnum90           1307 non-null int64
    auth_contactnum30           1307 non-null int64
    mobile_other_devices_cnt    1307 non-null int64
    idcard_other_devices_cnt    1307 non-null int64
    other_devices_cnt           1307 non-null int64
    overdue_count               1307 non-null int64
    is_overdue                  1307 non-null int64
    dtypes: float64(1), int64(9)
    memory usage: 102.2 KB
    


```python
magic_wand['user_id'] = magic_wand['user_id'].astype('str')
```

## <span id='3'>3、Compute corrcoef & IV value</span>

Merge the DataFrames with all kinds of features to get the BIG df
- dial_ratio_df: dial_ratio
- dial_maxmin_df: dial_maxmin
- dial_std_df: dial_std
- duration_df: duration_10_ratio,	duration_5_ratio
- mobile_bill_df: max_fee, min_fee, mean_fee
- mobile_basic_df: open_year, level, availableBalance, level_per_year
- city_number_df: city_num
- user_device_df: device_type, **mark**
- contact_num: contact_num
- frequent_contact_num: frequent_contact_num
- contact_1_times: contact_1_times
- contact_2_times: contact_2_times
- poweroff_df: poweroff_days, poweroff_times
- tongdun_df: age, tongdun_score, loan_7_num, loan_30_num
- xinyan_df: xinyan_score
- magic_wand: black_score, auth_contactnum180, auth_contactnum90, auth_contactnum30, mobile_other_devices_cnt, idcard_other_devices_cnt, other_devices_cnt, overdue_count, is_overdue


```python
df = pd.merge(user_device_df, dial_ratio_df, how='left', on='user_id')
df = pd.merge(df, dial_maxmin_df, how='left', on='user_id')
df = pd.merge(df, dial_std_df, how='left', on='user_id')
df = pd.merge(df, duration_df, how='left', on='user_id')
df = pd.merge(df, mobile_bill_df, how='left', on='user_id')
df = pd.merge(df, mobile_basic_df, how='left', on='user_id')
df = pd.merge(df, city_number_df, how='left', on='user_id')
df = pd.merge(df, contact_num, how='left', on='user_id')
df = pd.merge(df, frequent_contact_num, how='left', on='user_id')
df = pd.merge(df, contact_1_times, how='left', on='user_id')
df = pd.merge(df, contact_2_times, how='left', on='user_id')
df = pd.merge(df, poweroff_df, how='left', on='user_id')
df = pd.merge(df, tongdun_df, how='left', on='user_id')
df = pd.merge(df, xinyan_df, how='left', on='user_id')
df = pd.merge(df, magic_wand, how='left', on='user_id')
print(df.shape)
df.head()
```

    (1691, 39)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>device_type</th>
      <th>mark</th>
      <th>dial_ratio</th>
      <th>dial_maxmin</th>
      <th>dial_std</th>
      <th>count</th>
      <th>count_10</th>
      <th>count_5</th>
      <th>duration_10_ratio</th>
      <th>...</th>
      <th>xinyan_score</th>
      <th>black_score</th>
      <th>auth_contactnum180</th>
      <th>auth_contactnum90</th>
      <th>auth_contactnum30</th>
      <th>mobile_other_devices_cnt</th>
      <th>idcard_other_devices_cnt</th>
      <th>other_devices_cnt</th>
      <th>overdue_count</th>
      <th>is_overdue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15497</td>
      <td>2</td>
      <td>0</td>
      <td>0.307573</td>
      <td>0.396644</td>
      <td>0.137047</td>
      <td>647.0</td>
      <td>16.0</td>
      <td>47.0</td>
      <td>0.024730</td>
      <td>...</td>
      <td>0.0</td>
      <td>91.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27232</td>
      <td>2</td>
      <td>0</td>
      <td>0.473896</td>
      <td>0.261487</td>
      <td>0.095122</td>
      <td>1245.0</td>
      <td>12.0</td>
      <td>54.0</td>
      <td>0.009639</td>
      <td>...</td>
      <td>0.0</td>
      <td>63.0</td>
      <td>9.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33118</td>
      <td>2</td>
      <td>0</td>
      <td>0.468085</td>
      <td>0.221053</td>
      <td>0.081133</td>
      <td>564.0</td>
      <td>18.0</td>
      <td>42.0</td>
      <td>0.031915</td>
      <td>...</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35223</td>
      <td>1</td>
      <td>0</td>
      <td>0.429095</td>
      <td>0.155856</td>
      <td>0.050688</td>
      <td>818.0</td>
      <td>6.0</td>
      <td>18.0</td>
      <td>0.007335</td>
      <td>...</td>
      <td>0.0</td>
      <td>78.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36596</td>
      <td>2</td>
      <td>0</td>
      <td>0.438121</td>
      <td>0.109386</td>
      <td>0.040254</td>
      <td>2214.0</td>
      <td>20.0</td>
      <td>94.0</td>
      <td>0.009033</td>
      <td>...</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1691 entries, 0 to 1690
    Data columns (total 39 columns):
    user_id                     1691 non-null object
    device_type                 1691 non-null object
    mark                        1691 non-null object
    dial_ratio                  1689 non-null float64
    dial_maxmin                 1689 non-null float64
    dial_std                    1688 non-null float64
    count                       1689 non-null float64
    count_10                    1654 non-null float64
    count_5                     1687 non-null float64
    duration_10_ratio           1654 non-null float64
    duration_5_ratio            1687 non-null float64
    max_fee                     1691 non-null int64
    min_fee                     1691 non-null int64
    mean_fee                    1691 non-null float64
    open_year                   1572 non-null float64
    level                       1526 non-null float64
    availableBalance            1676 non-null float64
    level_per_year              1456 non-null float64
    city_num                    1689 non-null float64
    contact_num                 1687 non-null float64
    frequent_contact_num        1675 non-null float64
    contact_1_times             1675 non-null float64
    contact_2_times             1676 non-null float64
    poweroff_days               1326 non-null float64
    poweroff_times              1326 non-null float64
    age                         1691 non-null int64
    tongdun_score               1691 non-null int64
    loan_7_num                  1630 non-null float64
    loan_30_num                 1676 non-null float64
    xinyan_score                514 non-null float64
    black_score                 1307 non-null float64
    auth_contactnum180          1307 non-null float64
    auth_contactnum90           1307 non-null float64
    auth_contactnum30           1307 non-null float64
    mobile_other_devices_cnt    1307 non-null float64
    idcard_other_devices_cnt    1307 non-null float64
    other_devices_cnt           1307 non-null float64
    overdue_count               1307 non-null float64
    is_overdue                  1307 non-null float64
    dtypes: float64(32), int64(4), object(3)
    memory usage: 528.4+ KB
    


```python
df = df.convert_objects(convert_numeric=True)
```

    C:\Users\thunderbang\AppData\Local\Continuum\anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: convert_objects is deprecated.  To re-infer data dtypes for object columns, use DataFrame.infer_objects()
    For all other conversions use the data-type specific converters pd.to_datetime, pd.to_timedelta and pd.to_numeric.
      """Entry point for launching an IPython kernel.
    


```python
df.to_csv('Feature Extraction-0924.csv')
```


```python
df.drop('user_id', axis=1, inplace=True)
```


```python
# check missing data ratio
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>xinyan_score</th>
      <td>1177</td>
      <td>0.696038</td>
    </tr>
    <tr>
      <th>is_overdue</th>
      <td>384</td>
      <td>0.227085</td>
    </tr>
    <tr>
      <th>other_devices_cnt</th>
      <td>384</td>
      <td>0.227085</td>
    </tr>
    <tr>
      <th>idcard_other_devices_cnt</th>
      <td>384</td>
      <td>0.227085</td>
    </tr>
    <tr>
      <th>mobile_other_devices_cnt</th>
      <td>384</td>
      <td>0.227085</td>
    </tr>
    <tr>
      <th>auth_contactnum30</th>
      <td>384</td>
      <td>0.227085</td>
    </tr>
    <tr>
      <th>auth_contactnum90</th>
      <td>384</td>
      <td>0.227085</td>
    </tr>
    <tr>
      <th>auth_contactnum180</th>
      <td>384</td>
      <td>0.227085</td>
    </tr>
    <tr>
      <th>black_score</th>
      <td>384</td>
      <td>0.227085</td>
    </tr>
    <tr>
      <th>overdue_count</th>
      <td>384</td>
      <td>0.227085</td>
    </tr>
    <tr>
      <th>poweroff_times</th>
      <td>365</td>
      <td>0.215849</td>
    </tr>
    <tr>
      <th>poweroff_days</th>
      <td>365</td>
      <td>0.215849</td>
    </tr>
    <tr>
      <th>level_per_year</th>
      <td>235</td>
      <td>0.138971</td>
    </tr>
    <tr>
      <th>level</th>
      <td>165</td>
      <td>0.097575</td>
    </tr>
    <tr>
      <th>open_year</th>
      <td>119</td>
      <td>0.070373</td>
    </tr>
    <tr>
      <th>loan_7_num</th>
      <td>61</td>
      <td>0.036073</td>
    </tr>
    <tr>
      <th>duration_10_ratio</th>
      <td>37</td>
      <td>0.021881</td>
    </tr>
    <tr>
      <th>count_10</th>
      <td>37</td>
      <td>0.021881</td>
    </tr>
    <tr>
      <th>contact_1_times</th>
      <td>16</td>
      <td>0.009462</td>
    </tr>
    <tr>
      <th>frequent_contact_num</th>
      <td>16</td>
      <td>0.009462</td>
    </tr>
    <tr>
      <th>contact_2_times</th>
      <td>15</td>
      <td>0.008870</td>
    </tr>
    <tr>
      <th>availableBalance</th>
      <td>15</td>
      <td>0.008870</td>
    </tr>
    <tr>
      <th>loan_30_num</th>
      <td>15</td>
      <td>0.008870</td>
    </tr>
    <tr>
      <th>count_5</th>
      <td>4</td>
      <td>0.002365</td>
    </tr>
    <tr>
      <th>duration_5_ratio</th>
      <td>4</td>
      <td>0.002365</td>
    </tr>
    <tr>
      <th>contact_num</th>
      <td>4</td>
      <td>0.002365</td>
    </tr>
    <tr>
      <th>dial_std</th>
      <td>3</td>
      <td>0.001774</td>
    </tr>
    <tr>
      <th>city_num</th>
      <td>2</td>
      <td>0.001183</td>
    </tr>
    <tr>
      <th>count</th>
      <td>2</td>
      <td>0.001183</td>
    </tr>
    <tr>
      <th>dial_maxmin</th>
      <td>2</td>
      <td>0.001183</td>
    </tr>
    <tr>
      <th>dial_ratio</th>
      <td>2</td>
      <td>0.001183</td>
    </tr>
    <tr>
      <th>min_fee</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max_fee</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>mean_fee</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>tongdun_score</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>mark</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>device_type</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.corr()['mark'].sort_values()
```




    open_year                  -0.206789
    black_score                -0.183726
    device_type                -0.097995
    frequent_contact_num       -0.096326
    contact_num                -0.090697
    age                        -0.087685
    level                      -0.068064
    tongdun_score              -0.066402
    max_fee                    -0.031882
    mean_fee                   -0.023905
    availableBalance           -0.009738
    duration_10_ratio          -0.007429
    contact_1_times            -0.000745
    min_fee                     0.002950
    duration_5_ratio            0.016635
    city_num                    0.025281
    count_10                    0.032075
    loan_30_num                 0.038680
    count                       0.038824
    loan_7_num                  0.042033
    count_5                     0.050820
    contact_2_times             0.075215
    dial_ratio                  0.084610
    poweroff_times              0.098717
    overdue_count               0.106166
    poweroff_days               0.106527
    mobile_other_devices_cnt    0.107098
    other_devices_cnt           0.107098
    idcard_other_devices_cnt    0.112035
    level_per_year              0.112301
    is_overdue                  0.116463
    dial_maxmin                 0.118895
    dial_std                    0.132233
    auth_contactnum180          0.133430
    auth_contactnum30           0.150146
    auth_contactnum90           0.160175
    xinyan_score                0.340338
    mark                        1.000000
    Name: mark, dtype: float64




```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
fig, ax = plt.subplots(figsize = (10, 8))
sns.heatmap(df.corr(), ax=ax)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d26ab2d208>




![png](/assets/post9-2018-09-24/heatmap.png)



```python
iv(df, 'open_year', cut=2, n=4)
```

    woe:
     qcut
    (0.221, 2.662]   -0.766365
    (2.662, 4.804]   -0.057291
    (4.804, 7.56]     0.260760
    (7.56, 20.649]    0.465084
    Name: woe, dtype: float64
    iv:
     qcut
    (0.221, 2.662]    0.124405
    (2.662, 4.804]    0.000760
    (4.804, 7.56]     0.015977
    (7.56, 20.649]    0.050861
    Name: iv, dtype: float64
    




    (0.19200278567868745, 4)




```python
iv(df, 'xinyan_score')
```

    woe:
     xinyan_score
    0.0    0.307496
    1.0   -1.690239
    Name: woe, dtype: float64
    iv:
     xinyan_score
    0.0    0.021839
    1.0    0.161998
    Name: iv, dtype: float64
    




    0.1838369032455441




```python
iv(df, 'contact_1_times', cut=2, n=4)
```

    woe:
     qcut
    (-0.001, 20.0]     0.302628
    (20.0, 72.0]      -0.025003
    (72.0, 192.0]     -0.200259
    (192.0, 2008.0]   -0.062783
    Name: woe, dtype: float64
    iv:
     qcut
    (-0.001, 20.0]     0.023343
    (20.0, 72.0]       0.000154
    (72.0, 192.0]      0.009686
    (192.0, 2008.0]    0.000970
    Name: iv, dtype: float64
    




    (0.03415329169686423, 4)




```python
iv(df, 'city_num', cut=2, n=4)
```

    woe:
     qcut
    (0.999, 48.0]   -0.132301
    (48.0, 72.0]     0.269657
    (72.0, 96.0]     0.129630
    (96.0, 198.0]   -0.299074
    Name: woe, dtype: float64
    iv:
     qcut
    (0.999, 48.0]    0.004387
    (48.0, 72.0]     0.018654
    (72.0, 96.0]     0.004292
    (96.0, 198.0]    0.020767
    Name: iv, dtype: float64
    




    (0.04810102381440222, 4)



## <span id='4'>4、Reflection and summary</span>

本文重写了特征提取的全部代码，主要包括运营商和多头数据，并计算了所列特征的相关系数和IV值。简单来看，模型选用的特征有一些coff和iv值都较小，而未选用的特征也有一些coff和iv值较大，后续应进一步分析旧特征去模和新特征入模工作。

反思：1、还有很多新特征值得挖掘，例如性别等；
      2、现有的特征可以重新组合，以及模型中对数处理的方式值得商榷。
     
总结：本文整理了模型特征方面的许多的问题，可用于相关技术人员参考。特征对模型影响的分析工作待后续补充。
