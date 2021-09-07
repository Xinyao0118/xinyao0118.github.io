---
layout:     post
title:      Identify Fraudulent Activities 	
subtitle:   Use Random Forest to detect fraudulent activities for E-commerce websites.
date:       2021-04-28 	
author:     Xinyao Wu
header-img: img/post-bg-ioses.jpg
catalog: true
tags:
    - E-Commerce
    - Machine Learning
    - Random Forest
    - Supervised Learning
    - ROC Curve
---
# Goal
Build a machine learning model that predicts the probability that the first transaction of a new user is fraudulent.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, classification_report

import h2o
from h2o.frame import H2OFrame
from h2o.estimators.random_forest import H2ORandomForestEstimator

%matplotlib inline
```


```python
data = pd.read_csv('Fraud_Data.csv',parse_dates=['signup_time', 'purchase_time'])
```

## 1. Map users' IP addresses to their countries
```python

address2country = pd.read_csv('./IpAddress_to_Country.csv')
countries = []
for i in range(len(data)):
    ip_address = data.loc[i, 'ip_address']
    tmp = address2country[(address2country['lower_bound_ip_address'] <= ip_address) &
                          (address2country['upper_bound_ip_address'] >= ip_address)]
    if len(tmp) == 1:
        countries.append(tmp['country'].values[0])
    else:
        countries.append('NA')

data['country'] = countries
```


```python
data.head()
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
      <th>signup_time</th>
      <th>purchase_time</th>
      <th>purchase_value</th>
      <th>device_id</th>
      <th>source</th>
      <th>browser</th>
      <th>sex</th>
      <th>age</th>
      <th>ip_address</th>
      <th>class</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22058</td>
      <td>2015-02-24 22:55:49</td>
      <td>2015-04-18 02:47:11</td>
      <td>34</td>
      <td>QVPSPJUOCKZAR</td>
      <td>SEO</td>
      <td>Chrome</td>
      <td>M</td>
      <td>39</td>
      <td>7.327584e+08</td>
      <td>0</td>
      <td>Japan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>333320</td>
      <td>2015-06-07 20:39:50</td>
      <td>2015-06-08 01:38:54</td>
      <td>16</td>
      <td>EOGFQPIZPYXFZ</td>
      <td>Ads</td>
      <td>Chrome</td>
      <td>F</td>
      <td>53</td>
      <td>3.503114e+08</td>
      <td>0</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1359</td>
      <td>2015-01-01 18:52:44</td>
      <td>2015-01-01 18:52:45</td>
      <td>15</td>
      <td>YSSKYOSJHPPLJ</td>
      <td>SEO</td>
      <td>Opera</td>
      <td>M</td>
      <td>53</td>
      <td>2.621474e+09</td>
      <td>1</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>3</th>
      <td>150084</td>
      <td>2015-04-28 21:13:25</td>
      <td>2015-05-04 13:54:50</td>
      <td>44</td>
      <td>ATGTXKYKUDUQN</td>
      <td>SEO</td>
      <td>Safari</td>
      <td>M</td>
      <td>41</td>
      <td>3.840542e+09</td>
      <td>0</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>221365</td>
      <td>2015-07-21 07:09:52</td>
      <td>2015-09-09 18:40:53</td>
      <td>39</td>
      <td>NAUITBZFJKHWW</td>
      <td>Ads</td>
      <td>Safari</td>
      <td>M</td>
      <td>45</td>
      <td>4.155831e+08</td>
      <td>0</td>
      <td>United States</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Feature Engineering
```python
#check time difference between purchase and register
time_diff = data['purchase_time'] - data['signup_time']
time_diff = time_diff.apply(lambda x: x.seconds)
data['time_diff'] = time_diff

# Check user number for unique devices
device_num = data[['user_id', 'device_id']].groupby('device_id').count().reset_index()
device_num = device_num.rename(columns={'user_id': 'device_num'})
data = data.merge(device_num, how='left', on='device_id')
ip_num = data[['user_id', 'ip_address']].groupby('ip_address').count().reset_index()
ip_num = ip_num.rename(columns={'user_id': 'ip_num'})
data = data.merge(ip_num, how='left', on='ip_address')
```


```python
data['signup_day'] = data['signup_time'].apply(lambda x: x.dayofweek)
data['signup_week'] = data['signup_time'].apply(lambda x: x.week)

# Purchase day and week
data['purchase_day'] = data['purchase_time'].apply(lambda x: x.dayofweek)
data['purchase_week'] = data['purchase_time'].apply(lambda x: x.week)
columns = ['signup_day', 'signup_week', 'purchase_day', 'purchase_week', 'purchase_value', 'source',
           'browser', 'sex', 'age', 'country', 'time_diff', 'device_num', 'ip_num', 'class']
data = data[columns]
```
# 3. Build Random Forest Model with H2O Frame

```python

# Initialize H2O cluster
h2o.init()
h2o.remove_all()

# Transform to H2O Frame, and make sure the target variable is categorical
h2o_df = H2OFrame(data)

for name in ['signup_day', 'purchase_day', 'source', 'browser', 'sex', 'country', 'class']:
    h2o_df[name] = h2o_df[name].asfactor()
# Split into 70% training and 30% test dataset
strat_split = h2o_df['class'].stratified_split(test_frac=0.3, seed=42)

train = h2o_df[strat_split == 'train']
test = h2o_df[strat_split == 'test']

# Define features and target
feature = ['signup_day', 'signup_week', 'purchase_day', 'purchase_week', 'purchase_value',
           'source', 'browser', 'sex', 'age', 'country', 'time_diff', 'device_num', 'ip_num']
target = 'class'
# Build random forest model
model = H2ORandomForestEstimator(balance_classes=True, ntrees=100, mtries=-1, stopping_rounds=5,
                                 stopping_metric='auc', score_each_iteration=True, seed=42)
model.train(x=feature, y=target, training_frame=train, validation_frame=test)
```

    Checking whether there is an H2O instance running at http://localhost:54321 ..... not found.
    Attempting to start a local H2O server...
      Java Version: openjdk version "15.0.2" 2021-01-19; OpenJDK Runtime Environment (build 15.0.2+7); OpenJDK 64-Bit Server VM (build 15.0.2+7, mixed mode, sharing)
      Starting server from /opt/anaconda3/lib/python3.8/site-packages/h2o/backend/bin/h2o.jar
      Ice root: /var/folders/yf/vgp_y5cn7c79tm73jsfjm3k00000gn/T/tmpchdaa8vj
      JVM stdout: /var/folders/yf/vgp_y5cn7c79tm73jsfjm3k00000gn/T/tmpchdaa8vj/h2o_mia_started_from_python.out
      JVM stderr: /var/folders/yf/vgp_y5cn7c79tm73jsfjm3k00000gn/T/tmpchdaa8vj/h2o_mia_started_from_python.err
      Server is running at http://127.0.0.1:54321
    Connecting to H2O server at http://127.0.0.1:54321 ... successful.



<div style="overflow:auto"><table style="width:50%"><tr><td>H2O_cluster_uptime:</td>
<td>02 secs</td></tr>
<tr><td>H2O_cluster_timezone:</td>
<td>America/Los_Angeles</td></tr>
<tr><td>H2O_data_parsing_timezone:</td>
<td>UTC</td></tr>
<tr><td>H2O_cluster_version:</td>
<td>3.32.1.6</td></tr>
<tr><td>H2O_cluster_version_age:</td>
<td>17 days </td></tr>
<tr><td>H2O_cluster_name:</td>
<td>H2O_from_python_mia_ykfcn6</td></tr>
<tr><td>H2O_cluster_total_nodes:</td>
<td>1</td></tr>
<tr><td>H2O_cluster_free_memory:</td>
<td>4 Gb</td></tr>
<tr><td>H2O_cluster_total_cores:</td>
<td>12</td></tr>
<tr><td>H2O_cluster_allowed_cores:</td>
<td>12</td></tr>
<tr><td>H2O_cluster_status:</td>
<td>accepting new members, healthy</td></tr>
<tr><td>H2O_connection_url:</td>
<td>http://127.0.0.1:54321</td></tr>
<tr><td>H2O_connection_proxy:</td>
<td>{"http": null, "https": null}</td></tr>
<tr><td>H2O_internal_security:</td>
<td>False</td></tr>
<tr><td>H2O_API_Extensions:</td>
<td>Amazon S3, XGBoost, Algos, AutoML, Core V3, TargetEncoder, Core V4</td></tr>
<tr><td>Python_version:</td>
<td>3.8.8 final</td></tr></table></div>


    Parse progress: |█████████████████████████████████████████████████████████| 100%
    drf Model Build progress: |███████████████████████████████████████████████| 100%


# 4. Show the feature importance
```python
# Feature importance
importance = model.varimp(use_pandas=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x='scaled_importance', y='variable', data=importance)
plt.show()
```


<img src="/img/output_7_0.png"/>


```python

# Make predictions
train_true = train.as_data_frame()['class'].values
test_true = test.as_data_frame()['class'].values
train_pred = model.predict(train).as_data_frame()['p1'].values
test_pred = model.predict(test).as_data_frame()['p1'].values

train_fpr, train_tpr, _ = roc_curve(train_true, train_pred)
test_fpr, test_tpr, _ = roc_curve(test_true, test_pred)
train_auc = np.round(auc(train_fpr, train_tpr), 3)
test_auc = np.round(auc(test_fpr, test_tpr), 3)
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    drf prediction progress: |████████████████████████████████████████████████| 100%



```python
# Classification report
print(classification_report(y_true=test_true, y_pred=(test_pred > 0.5).astype(int)))
```

                  precision    recall  f1-score   support

               0       0.95      1.00      0.98     41088
               1       1.00      0.53      0.69      4245

        accuracy                           0.96     45333
       macro avg       0.98      0.76      0.83     45333
    weighted avg       0.96      0.96      0.95     45333



### Explanation
class = 0 : not fraudulent \
class = 1 : fraudulent \
recall = 0.53 for class 1, meaning this model can only detect 53% of all fraudulent activities. \
precision = 0.95 for class 0 , meaning 95% the non-fraudulent activities defined by this model are real non-fraudulent activities. \
The reason why recall rate is low for fraudulent class is that the cut-off point is default to be 0.5. So I may lower the cut-off point to see the recall change.


```python
# Classification report
print(classification_report(y_true=test_true, y_pred=(test_pred > 0.05).astype(int)))
```

                  precision    recall  f1-score   support

               0       0.97      0.95      0.96     41088
               1       0.58      0.67      0.62      4245

        accuracy                           0.92     45333
       macro avg       0.77      0.81      0.79     45333
    weighted avg       0.93      0.92      0.93     45333



### Explanation
The recall value of fraudulent class increased to 0.67 after decreasing cut-off point from 0.5 to 0.05. \
However, other metrics performs worse than before. \
For example, the precision for fraudulent class decreased from1 to 0.58, meaning that 57% of the detedt fraudulent are real fraudulent activities. \
A lower presicion obviously is not what I want, so for further research, hyperparatemer cut-off point should be tuned subtily for best classification. \
Here is a way to evaluate the model's classification ability, which are ROC curve and AUC value.


```python

train_fpr = np.insert(train_fpr, 0, 0)
train_tpr = np.insert(train_tpr, 0, 0)
test_fpr = np.insert(test_fpr, 0, 0)
test_tpr = np.insert(test_tpr, 0, 0)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(train_fpr, train_tpr, label='Train AUC: ' + str(train_auc))
ax.plot(test_fpr, test_tpr, label='Test AUC: ' + str(test_auc))
ax.plot(train_fpr, train_fpr, 'k--', label='Chance Curve')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.grid(True)
ax.legend(fontsize=12)
plt.show()
```

<img src="/img/output_13_0.png"/>


# 5. Conclusion

The Test AUC score is 0.85. \
Normally for test AUC score 0.7 to 0.8 is considered acceptable, 0.8 to 0.9 is considered excellent.
So this cluster is excellent for detecting fraudulent activities.


```python
h2o.cluster().shutdown()
```
