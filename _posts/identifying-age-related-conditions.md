---
layout:     post
title:      Identifying age related conditions
subtitle:   Utilized
date:       2023-07-22
author:     Xinyao Wu
header-img: img/home-bg.jpg
catalog: true
tags:
    - Machine Learning
    - Healthcare
    - LightGBM
    - disease prediction
    - AI
---
```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/icr-identify-age-related-conditions/sample_submission.csv
    /kaggle/input/icr-identify-age-related-conditions/greeks.csv
    /kaggle/input/icr-identify-age-related-conditions/train.csv
    /kaggle/input/icr-identify-age-related-conditions/test.csv



```python
import gc
import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt

np.random.seed(2)
```


```python
def reduce_mem_usage(df, verbose=False):
    """
    Utility function to reduce the memory usage of pandas dataframes

    Parameters
    ----------
    df: pandas.Dataframe
    verbose: Boolean
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
        if col_type == 'object' or col_type.name == 'category':
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
```


```python
def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df,verbose=True)
    return df
```


```python
train_data = import_data("/kaggle/input/icr-identify-age-related-conditions/train.csv")
```

    Mem. usage decreased to  0.15 Mb (44.2% reduction)



```python
train_data['cat_EJ'] = train_data['EJ'].cat.codes
```


```python
feat_names = ['AB',
 'AF',
 'BC',
 'BQ',
 'CC',
 'CD ',
 'CR',
 'DA',
 'DE',
 'DH',
 'DL',
 'DN',
 'DU',
 'DY',
 'EB',
 'EE',
 'EH',
 'EL',
 'EP',
 'EU',
 'FE',
 'FI',
 'FL',
 'FR',
 'GL']
col_names = feat_names+['Class'] # 特征 + label
```


```python
train_dset = lgb.Dataset(
    data=train_data[feat_names], # 训练集特征
    label=train_data["Class"].values, # 训练集label
    free_raw_data=False, # 关闭原始数据的内存占用
)
```


```python
# Optimization
from sklearn.model_selection import TimeSeriesSplit # 导入时间序列交叉验证
tscv = TimeSeriesSplit(5) # 5折时间序列交叉验证
for fold, (trn_ind, val_ind) in enumerate(tscv.split(train_data)):
    # print(f"train length: {len(trn_ind)}, valid length: {len(val_ind)}")
    train_df = train_data.loc[trn_ind, :] # 训练集
    valid_df = train_data.loc[val_ind, :] # 验证集

print(f"train_df.shape:{train_df.shape}; valid_df.shape:{valid_df.shape}")
```

    train_df.shape:(515, 59); valid_df.shape:(102, 59)



```python
train_dset = lgb.Dataset(
    data=train_df[feat_names], # 训练集特征
    label=train_df["Class"].values, # 训练集label
    free_raw_data=False, # 关闭原始数据的内存占用
)

valid_dset = lgb.Dataset(
    data=valid_df[feat_names], # 验证集特征
    label=valid_df["Class"].values, # 验证集label
    free_raw_data=False, # 关闭原始数据的内存占用
)

#del train_data
#del train_df
#del valid_df
#gc.collect()
```


```python
model_params = {
    'boosting': 'gbdt', # dart 提升算法
    'objective': 'binary', # 均方误差
    'metric': 'binary_logloss', # # 均方根误差
    'learning_rate': 0.0001,  # 学习率
    'seed': 42, # 随机种子\
}
```


```python
import optuna # 导入optuna
def objective(trial):
    model_params = {
        'boosting': 'gbdt', # dart 提升算法
        'objective': 'binary', # 均方误差
        'metric': 'auc', # # 均方根误差
        'learning_rate':  trial.suggest_loguniform("learning_rate", 0.005, 0.1),  # 学习率
        'seed': 42, # 随机种子,
         'num_leaves': trial.suggest_int("num_leaves", 5, 20), # 最大叶子数量
        'max_bin': trial.suggest_int("max_bin", 2, 30), # 最大分箱数
        'feature_fraction': trial.suggest_discrete_uniform("feature_fraction", 0.05, 0.5, 0.1), # 特征采样比例
    }

    _model_params = dict(model_params)
    _model_params["seed"] = 42 # 随机种子

    log_callback = lgb.log_evaluation(period=20) # 训练日志频率

    model = lgb.train(
        params=_model_params, # 参数
        train_set=train_dset, # 训练集
        valid_sets=[train_dset, valid_dset], # 验证集
        #feval=pearsonr, # 评估函数
        callbacks=[log_callback,], # 训练日志
    )

    #lgb.plot_importance(model, figsize=(8,15), importance_type="split", max_num_features=30) # split 特征重要度
    #lgb.plot_importance(model, figsize=(8,15), importance_type="gain", max_num_features=30) # gain 特征重要度
    #plt.show()

    return model.best_score["valid_1"]['auc']# best score
```


```python
study = optuna.create_study(direction='maximize') # 创建study，最大化score
study.optimize(objective, n_trials=100) # 进行100次试验
```

    [I 2023-08-02 04:49:51,306] A new study created in memory with name: no-name-2aa7173a-eebb-493f-adde-c243031bba9e
    /tmp/ipykernel_32/3356471658.py:7: FutureWarning: suggest_loguniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float(..., log=True) instead.
      'learning_rate':  trial.suggest_loguniform("learning_rate", 0.005, 0.1),  # 学习率
    /tmp/ipykernel_32/3356471658.py:11: FutureWarning: suggest_discrete_uniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float(..., step=...) instead.
      'feature_fraction': trial.suggest_discrete_uniform("feature_fraction", 0.05, 0.5, 0.1), # 特征采样比例
    /opt/conda/lib/python3.10/site-packages/optuna/distributions.py:685: UserWarning: The distribution is specified by [0.05, 0.5] and step=0.1, but the range is not divisible by `step`. It will be replaced by [0.05, 0.45].
      warnings.warn(


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000611 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 175
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.984475	valid_1's auc: 0.972318
    [40]	training's auc: 0.992614	valid_1's auc: 0.979239
    [60]	training's auc: 0.996786	valid_1's auc: 0.984775
    [80]	training's auc: 0.9993	valid_1's auc: 0.984083


    [I 2023-08-02 04:49:51,583] Trial 0 finished with value: 0.986159169550173 and parameters: {'learning_rate': 0.04213406914172991, 'num_leaves': 12, 'max_bin': 7, 'feature_fraction': 0.35000000000000003}. Best is trial 0 with value: 0.986159169550173.
    /tmp/ipykernel_32/3356471658.py:7: FutureWarning: suggest_loguniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float(..., log=True) instead.
      'learning_rate':  trial.suggest_loguniform("learning_rate", 0.005, 0.1),  # 学习率
    /tmp/ipykernel_32/3356471658.py:11: FutureWarning: suggest_discrete_uniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float(..., step=...) instead.
      'feature_fraction': trial.suggest_discrete_uniform("feature_fraction", 0.05, 0.5, 0.1), # 特征采样比例
    /opt/conda/lib/python3.10/site-packages/optuna/distributions.py:685: UserWarning: The distribution is specified by [0.05, 0.5] and step=0.1, but the range is not divisible by `step`. It will be replaced by [0.05, 0.45].
      warnings.warn(
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:51,753] Trial 1 finished with value: 0.9868512110726644 and parameters: {'learning_rate': 0.058895605936283575, 'num_leaves': 9, 'max_bin': 27, 'feature_fraction': 0.25}. Best is trial 1 with value: 0.9868512110726644.


    [100]	training's auc: 0.999896	valid_1's auc: 0.986159
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001204 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 675
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.989866	valid_1's auc: 0.96955
    [40]	training's auc: 0.998393	valid_1's auc: 0.982699
    [60]	training's auc: 0.99987	valid_1's auc: 0.986159
    [80]	training's auc: 1	valid_1's auc: 0.985467
    [100]	training's auc: 1	valid_1's auc: 0.986851
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000524 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 300
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:52,011] Trial 2 finished with value: 0.9854671280276817 and parameters: {'learning_rate': 0.007957118224840812, 'num_leaves': 17, 'max_bin': 12, 'feature_fraction': 0.25}. Best is trial 1 with value: 0.9868512110726644.


    [20]	training's auc: 0.981832	valid_1's auc: 0.974394
    [40]	training's auc: 0.99124	valid_1's auc: 0.977855
    [60]	training's auc: 0.99251	valid_1's auc: 0.979931
    [80]	training's auc: 0.993909	valid_1's auc: 0.982007
    [100]	training's auc: 0.994817	valid_1's auc: 0.985467


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000547 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 350
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.986108	valid_1's auc: 0.972318
    [40]	training's auc: 0.992458	valid_1's auc: 0.975779
    [60]	training's auc: 0.994169	valid_1's auc: 0.979239
    [80]	training's auc: 0.995801	valid_1's auc: 0.981315


    [I 2023-08-02 04:49:52,251] Trial 3 finished with value: 0.9840830449826989 and parameters: {'learning_rate': 0.010897503610088453, 'num_leaves': 15, 'max_bin': 14, 'feature_fraction': 0.25}. Best is trial 1 with value: 0.9868512110726644.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:52,421] Trial 4 finished with value: 0.9916955017301038 and parameters: {'learning_rate': 0.04769812119350733, 'num_leaves': 9, 'max_bin': 23, 'feature_fraction': 0.25}. Best is trial 4 with value: 0.9916955017301038.


    [100]	training's auc: 0.997097	valid_1's auc: 0.984083
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000570 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 575
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.986367	valid_1's auc: 0.977163
    [40]	training's auc: 0.995827	valid_1's auc: 0.987543
    [60]	training's auc: 0.998989	valid_1's auc: 0.988927
    [80]	training's auc: 0.999896	valid_1's auc: 0.990311
    [100]	training's auc: 1	valid_1's auc: 0.991696
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000577 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 375
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:52,657] Trial 5 finished with value: 0.9937716262975779 and parameters: {'learning_rate': 0.09933171604533854, 'num_leaves': 14, 'max_bin': 15, 'feature_fraction': 0.45}. Best is trial 5 with value: 0.9937716262975779.


    [20]	training's auc: 0.998886	valid_1's auc: 0.988235
    [40]	training's auc: 1	valid_1's auc: 0.986159
    [60]	training's auc: 1	valid_1's auc: 0.991696
    [80]	training's auc: 1	valid_1's auc: 0.992388
    [100]	training's auc: 1	valid_1's auc: 0.993772
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000543 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 675
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:52,807] Trial 6 finished with value: 0.9840830449826989 and parameters: {'learning_rate': 0.015822443396750974, 'num_leaves': 8, 'max_bin': 27, 'feature_fraction': 0.15000000000000002}. Best is trial 5 with value: 0.9937716262975779.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [20]	training's auc: 0.984164	valid_1's auc: 0.93564
    [40]	training's auc: 0.988726	valid_1's auc: 0.967474
    [60]	training's auc: 0.99168	valid_1's auc: 0.97301
    [80]	training's auc: 0.993443	valid_1's auc: 0.982699
    [100]	training's auc: 0.994454	valid_1's auc: 0.984083
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000551 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 400
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [20]	training's auc: 0.961953	valid_1's auc: 0.854671
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf


    [I 2023-08-02 04:49:53,038] Trial 7 finished with value: 0.946712802768166 and parameters: {'learning_rate': 0.005120083817605247, 'num_leaves': 18, 'max_bin': 16, 'feature_fraction': 0.05}. Best is trial 5 with value: 0.9937716262975779.


    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [40]	training's auc: 0.973305	valid_1's auc: 0.888581
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [60]	training's auc: 0.97372	valid_1's auc: 0.921107
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [80]	training's auc: 0.973746	valid_1's auc: 0.928028
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [100]	training's auc: 0.976182	valid_1's auc: 0.946713
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000380 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 4
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 2
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [20]	training's auc: 0.530933	valid_1's auc: 0.557785
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [40]	training's auc: 0.530933	valid_1's auc: 0.557785
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [60]	training's auc: 0.530933	valid_1's auc: 0.557785
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [80]	training's auc: 0.530933	valid_1's auc: 0.557785
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:53,114] Trial 8 finished with value: 0.5577854671280277 and parameters: {'learning_rate': 0.021162108109272493, 'num_leaves': 10, 'max_bin': 2, 'feature_fraction': 0.15000000000000002}. Best is trial 5 with value: 0.9937716262975779.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:53,269] Trial 9 finished with value: 0.9910034602076124 and parameters: {'learning_rate': 0.03832618108526671, 'num_leaves': 8, 'max_bin': 14, 'feature_fraction': 0.25}. Best is trial 5 with value: 0.9937716262975779.


    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [100]	training's auc: 0.530933	valid_1's auc: 0.557785
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000633 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 350
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.972683	valid_1's auc: 0.972318
    [40]	training's auc: 0.988415	valid_1's auc: 0.975087
    [60]	training's auc: 0.994039	valid_1's auc: 0.984083
    [80]	training's auc: 0.997849	valid_1's auc: 0.987543
    [100]	training's auc: 0.999119	valid_1's auc: 0.991003
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000531 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 525
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [20]	training's auc: 0.998782	valid_1's auc: 0.975779
    [40]	training's auc: 1	valid_1's auc: 0.984083
    [60]	training's auc: 1	valid_1's auc: 0.987543


    [I 2023-08-02 04:49:53,605] Trial 10 finished with value: 0.9875432525951557 and parameters: {'learning_rate': 0.08581695661211744, 'num_leaves': 20, 'max_bin': 21, 'feature_fraction': 0.45}. Best is trial 5 with value: 0.9937716262975779.


    [80]	training's auc: 1	valid_1's auc: 0.988235
    [100]	training's auc: 1	valid_1's auc: 0.987543
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000620 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 525
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.984164	valid_1's auc: 0.968858
    [40]	training's auc: 0.995335	valid_1's auc: 0.984775
    [60]	training's auc: 0.999248	valid_1's auc: 0.986159
    [80]	training's auc: 0.999948	valid_1's auc: 0.987543
    [100]	training's auc: 1	valid_1's auc: 0.988235


    [I 2023-08-02 04:49:53,743] Trial 11 finished with value: 0.9882352941176471 and parameters: {'learning_rate': 0.09746279862177851, 'num_leaves': 5, 'max_bin': 21, 'feature_fraction': 0.45}. Best is trial 5 with value: 0.9937716262975779.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000626 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 525
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.994972	valid_1's auc: 0.974394
    [40]	training's auc: 0.999456	valid_1's auc: 0.982699
    [60]	training's auc: 1	valid_1's auc: 0.986159
    [80]	training's auc: 1	valid_1's auc: 0.987543


    [I 2023-08-02 04:49:53,984] Trial 12 finished with value: 0.9882352941176471 and parameters: {'learning_rate': 0.05791668710076641, 'num_leaves': 13, 'max_bin': 21, 'feature_fraction': 0.35000000000000003}. Best is trial 5 with value: 0.9937716262975779.


    [100]	training's auc: 1	valid_1's auc: 0.988235
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000576 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 575
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.998549	valid_1's auc: 0.977855
    [40]	training's auc: 1	valid_1's auc: 0.987543
    [60]	training's auc: 1	valid_1's auc: 0.991003


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:54,234] Trial 13 finished with value: 0.9944636678200692 and parameters: {'learning_rate': 0.09411790981958434, 'num_leaves': 13, 'max_bin': 23, 'feature_fraction': 0.35000000000000003}. Best is trial 13 with value: 0.9944636678200692.


    [80]	training's auc: 1	valid_1's auc: 0.991696
    [100]	training's auc: 1	valid_1's auc: 0.994464
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000562 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 225
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.995283	valid_1's auc: 0.992388
    [40]	training's auc: 0.999948	valid_1's auc: 0.992388


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:54,490] Trial 14 finished with value: 0.9944636678200692 and parameters: {'learning_rate': 0.08289795726554436, 'num_leaves': 14, 'max_bin': 9, 'feature_fraction': 0.35000000000000003}. Best is trial 13 with value: 0.9944636678200692.


    [60]	training's auc: 1	valid_1's auc: 0.99308
    [80]	training's auc: 1	valid_1's auc: 0.994464
    [100]	training's auc: 1	valid_1's auc: 0.994464
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000550 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 175
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [20]	training's auc: 0.984216	valid_1's auc: 0.97301


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:54,762] Trial 15 finished with value: 0.9813148788927336 and parameters: {'learning_rate': 0.03171605242865897, 'num_leaves': 16, 'max_bin': 7, 'feature_fraction': 0.35000000000000003}. Best is trial 13 with value: 0.9944636678200692.


    [40]	training's auc: 0.991732	valid_1's auc: 0.975087
    [60]	training's auc: 0.995724	valid_1's auc: 0.981315
    [80]	training's auc: 0.998004	valid_1's auc: 0.980623
    [100]	training's auc: 0.999378	valid_1's auc: 0.981315
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000535 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 750
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:54,985] Trial 16 finished with value: 0.9916955017301038 and parameters: {'learning_rate': 0.0688655620050642, 'num_leaves': 11, 'max_bin': 30, 'feature_fraction': 0.35000000000000003}. Best is trial 13 with value: 0.9944636678200692.


    [20]	training's auc: 0.996138	valid_1's auc: 0.972318
    [40]	training's auc: 0.999767	valid_1's auc: 0.983391
    [60]	training's auc: 1	valid_1's auc: 0.990311
    [80]	training's auc: 1	valid_1's auc: 0.988927
    [100]	training's auc: 1	valid_1's auc: 0.991696
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000584 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 250
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:55,233] Trial 17 finished with value: 0.9910034602076124 and parameters: {'learning_rate': 0.07039463150248246, 'num_leaves': 13, 'max_bin': 10, 'feature_fraction': 0.35000000000000003}. Best is trial 13 with value: 0.9944636678200692.


    [20]	training's auc: 0.994687	valid_1's auc: 0.988927
    [40]	training's auc: 0.999611	valid_1's auc: 0.987543
    [60]	training's auc: 1	valid_1's auc: 0.99308
    [80]	training's auc: 1	valid_1's auc: 0.991696
    [100]	training's auc: 1	valid_1's auc: 0.991003


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000607 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 100
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [20]	training's auc: 0.98051	valid_1's auc: 0.945329
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [40]	training's auc: 0.989348	valid_1's auc: 0.961246
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [60]	training's auc: 0.994013	valid_1's auc: 0.964014
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf


    [I 2023-08-02 04:49:55,542] Trial 18 finished with value: 0.9688581314878892 and parameters: {'learning_rate': 0.030299285753333207, 'num_leaves': 19, 'max_bin': 4, 'feature_fraction': 0.45}. Best is trial 13 with value: 0.9944636678200692.


    [80]	training's auc: 0.996683	valid_1's auc: 0.96609
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [100]	training's auc: 0.998419	valid_1's auc: 0.968858
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000656 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 475
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.996112	valid_1's auc: 0.948097
    [40]	training's auc: 0.999585	valid_1's auc: 0.968166


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:55,809] Trial 19 finished with value: 0.9854671280276817 and parameters: {'learning_rate': 0.05189787525230193, 'num_leaves': 15, 'max_bin': 19, 'feature_fraction': 0.15000000000000002}. Best is trial 13 with value: 0.9944636678200692.


    [60]	training's auc: 1	valid_1's auc: 0.971626
    [80]	training's auc: 1	valid_1's auc: 0.986159
    [100]	training's auc: 1	valid_1's auc: 0.985467
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000611 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 275
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.978035	valid_1's auc: 0.968166
    [40]	training's auc: 0.988752	valid_1's auc: 0.970934
    [60]	training's auc: 0.994557	valid_1's auc: 0.983391


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:55,945] Trial 20 finished with value: 0.9840830449826989 and parameters: {'learning_rate': 0.07618064464781599, 'num_leaves': 5, 'max_bin': 11, 'feature_fraction': 0.35000000000000003}. Best is trial 13 with value: 0.9944636678200692.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [80]	training's auc: 0.997875	valid_1's auc: 0.984083
    [100]	training's auc: 0.999456	valid_1's auc: 0.984083
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000595 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 450
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [20]	training's auc: 0.998756	valid_1's auc: 0.985467
    [40]	training's auc: 1	valid_1's auc: 0.988927
    [60]	training's auc: 1	valid_1's auc: 0.991696


    [I 2023-08-02 04:49:56,220] Trial 21 finished with value: 0.9896193771626297 and parameters: {'learning_rate': 0.08988476593385698, 'num_leaves': 15, 'max_bin': 18, 'feature_fraction': 0.45}. Best is trial 13 with value: 0.9944636678200692.


    [80]	training's auc: 1	valid_1's auc: 0.993772
    [100]	training's auc: 1	valid_1's auc: 0.989619
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000278 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 200
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.996475	valid_1's auc: 0.980623


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:56,565] Trial 22 finished with value: 0.9903114186851211 and parameters: {'learning_rate': 0.09764331927831399, 'num_leaves': 14, 'max_bin': 8, 'feature_fraction': 0.45}. Best is trial 13 with value: 0.9944636678200692.


    [40]	training's auc: 1	valid_1's auc: 0.985467
    [60]	training's auc: 1	valid_1's auc: 0.989619
    [80]	training's auc: 1	valid_1's auc: 0.988927
    [100]	training's auc: 1	valid_1's auc: 0.990311


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:56,791] Trial 23 finished with value: 0.995847750865052 and parameters: {'learning_rate': 0.06803053801685711, 'num_leaves': 12, 'max_bin': 25, 'feature_fraction': 0.35000000000000003}. Best is trial 23 with value: 0.995847750865052.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000595 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.996605	valid_1's auc: 0.984775
    [40]	training's auc: 0.999844	valid_1's auc: 0.991696
    [60]	training's auc: 1	valid_1's auc: 0.994464
    [80]	training's auc: 1	valid_1's auc: 0.99654
    [100]	training's auc: 1	valid_1's auc: 0.995848
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000536 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.996683	valid_1's auc: 0.987543
    [40]	training's auc: 0.999793	valid_1's auc: 0.993772
    [60]	training's auc: 1	valid_1's auc: 0.993772
    [80]	training's auc: 1	valid_1's auc: 0.997232
    [100]	training's auc: 1	valid_1's auc: 0.995848


    [I 2023-08-02 04:49:57,023] Trial 24 finished with value: 0.995847750865052 and parameters: {'learning_rate': 0.07332176280191201, 'num_leaves': 11, 'max_bin': 25, 'feature_fraction': 0.35000000000000003}. Best is trial 23 with value: 0.995847750865052.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000652 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.995387	valid_1's auc: 0.988235
    [40]	training's auc: 0.999533	valid_1's auc: 0.993772
    [60]	training's auc: 1	valid_1's auc: 0.994464


    [I 2023-08-02 04:49:57,305] Trial 25 finished with value: 0.9951557093425606 and parameters: {'learning_rate': 0.06273555307323773, 'num_leaves': 11, 'max_bin': 25, 'feature_fraction': 0.35000000000000003}. Best is trial 23 with value: 0.995847750865052.


    [80]	training's auc: 1	valid_1's auc: 0.995156
    [100]	training's auc: 1	valid_1's auc: 0.995156
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000547 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 650
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.992847	valid_1's auc: 0.985467
    [40]	training's auc: 0.999119	valid_1's auc: 0.99308
    [60]	training's auc: 1	valid_1's auc: 0.99308


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:57,529] Trial 26 finished with value: 0.9937716262975779 and parameters: {'learning_rate': 0.06003086681743414, 'num_leaves': 11, 'max_bin': 26, 'feature_fraction': 0.35000000000000003}. Best is trial 23 with value: 0.995847750865052.


    [80]	training's auc: 1	valid_1's auc: 0.99308
    [100]	training's auc: 1	valid_1's auc: 0.993772
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000535 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 750
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.989633	valid_1's auc: 0.984775
    [40]	training's auc: 0.996812	valid_1's auc: 0.984083
    [60]	training's auc: 0.999145	valid_1's auc: 0.987543


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:57,735] Trial 27 finished with value: 0.9937716262975779 and parameters: {'learning_rate': 0.04498661938305263, 'num_leaves': 10, 'max_bin': 30, 'feature_fraction': 0.25}. Best is trial 23 with value: 0.995847750865052.


    [80]	training's auc: 0.999896	valid_1's auc: 0.992388
    [100]	training's auc: 1	valid_1's auc: 0.993772
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000587 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.986652	valid_1's auc: 0.984083
    [40]	training's auc: 0.995905	valid_1's auc: 0.984775
    [60]	training's auc: 0.999404	valid_1's auc: 0.984775
    [80]	training's auc: 1	valid_1's auc: 0.984775
    [100]	training's auc: 1	valid_1's auc: 0.989619


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:57,901] Trial 28 finished with value: 0.9896193771626297 and parameters: {'learning_rate': 0.06976157850078615, 'num_leaves': 7, 'max_bin': 24, 'feature_fraction': 0.35000000000000003}. Best is trial 23 with value: 0.995847750865052.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:58,131] Trial 29 finished with value: 0.9972318339100346 and parameters: {'learning_rate': 0.03997959493902798, 'num_leaves': 12, 'max_bin': 25, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000543 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.993443	valid_1's auc: 0.987543
    [40]	training's auc: 0.997693	valid_1's auc: 0.991003
    [60]	training's auc: 0.999585	valid_1's auc: 0.995848
    [80]	training's auc: 1	valid_1's auc: 0.997232
    [100]	training's auc: 1	valid_1's auc: 0.997232


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:58,355] Trial 30 finished with value: 0.9397923875432526 and parameters: {'learning_rate': 0.04084600888251869, 'num_leaves': 12, 'max_bin': 28, 'feature_fraction': 0.05}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000532 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 700
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [20]	training's auc: 0.973098	valid_1's auc: 0.882353
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [40]	training's auc: 0.987015	valid_1's auc: 0.901038
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [60]	training's auc: 0.988881	valid_1's auc: 0.914879
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [80]	training's auc: 0.991395	valid_1's auc: 0.913495
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [100]	training's auc: 0.993728	valid_1's auc: 0.939792


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:58,577] Trial 31 finished with value: 0.995847750865052 and parameters: {'learning_rate': 0.0528758883194643, 'num_leaves': 11, 'max_bin': 25, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000618 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.995257	valid_1's auc: 0.986851
    [40]	training's auc: 0.998808	valid_1's auc: 0.991696
    [60]	training's auc: 0.999922	valid_1's auc: 0.994464
    [80]	training's auc: 1	valid_1's auc: 0.99654
    [100]	training's auc: 1	valid_1's auc: 0.995848


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000535 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 700
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.994661	valid_1's auc: 0.977855
    [40]	training's auc: 0.999093	valid_1's auc: 0.981315
    [60]	training's auc: 0.999922	valid_1's auc: 0.99308
    [80]	training's auc: 1	valid_1's auc: 0.99308
    [100]	training's auc: 1	valid_1's auc: 0.99308


    [I 2023-08-02 04:49:58,817] Trial 32 finished with value: 0.9930795847750865 and parameters: {'learning_rate': 0.052875142582896405, 'num_leaves': 12, 'max_bin': 28, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:59,028] Trial 33 finished with value: 0.995847750865052 and parameters: {'learning_rate': 0.05190953317227754, 'num_leaves': 10, 'max_bin': 25, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000525 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.991214	valid_1's auc: 0.980623
    [40]	training's auc: 0.997797	valid_1's auc: 0.986851
    [60]	training's auc: 0.999611	valid_1's auc: 0.995156
    [80]	training's auc: 1	valid_1's auc: 0.99654
    [100]	training's auc: 1	valid_1's auc: 0.995848


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:59,254] Trial 34 finished with value: 0.9916955017301038 and parameters: {'learning_rate': 0.038190182562596243, 'num_leaves': 11, 'max_bin': 19, 'feature_fraction': 0.45}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000647 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 475
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.992899	valid_1's auc: 0.982699
    [40]	training's auc: 0.996579	valid_1's auc: 0.983391
    [60]	training's auc: 0.999015	valid_1's auc: 0.987543
    [80]	training's auc: 0.999844	valid_1's auc: 0.991003
    [100]	training's auc: 1	valid_1's auc: 0.991696


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:59,449] Trial 35 finished with value: 0.9944636678200692 and parameters: {'learning_rate': 0.059764018441050226, 'num_leaves': 9, 'max_bin': 23, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000548 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 575
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.988337	valid_1's auc: 0.979239
    [40]	training's auc: 0.997279	valid_1's auc: 0.988235
    [60]	training's auc: 0.999663	valid_1's auc: 0.994464
    [80]	training's auc: 1	valid_1's auc: 0.995848
    [100]	training's auc: 1	valid_1's auc: 0.994464
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000563 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 700
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [I 2023-08-02 04:49:59,694] Trial 36 finished with value: 0.9868512110726644 and parameters: {'learning_rate': 0.049611479798625284, 'num_leaves': 12, 'max_bin': 28, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [20]	training's auc: 0.99321	valid_1's auc: 0.961938
    [40]	training's auc: 0.999248	valid_1's auc: 0.975779
    [60]	training's auc: 0.99987	valid_1's auc: 0.982699
    [80]	training's auc: 1	valid_1's auc: 0.984083
    [100]	training's auc: 1	valid_1's auc: 0.986851
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000549 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 550
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:49:59,869] Trial 37 finished with value: 0.9910034602076124 and parameters: {'learning_rate': 0.07677039360983234, 'num_leaves': 7, 'max_bin': 22, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [20]	training's auc: 0.986627	valid_1's auc: 0.977855
    [40]	training's auc: 0.997797	valid_1's auc: 0.986851
    [60]	training's auc: 0.999689	valid_1's auc: 0.988927
    [80]	training's auc: 1	valid_1's auc: 0.990311
    [100]	training's auc: 1	valid_1's auc: 0.991003
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000572 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 650
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.989581	valid_1's auc: 0.975087


    [I 2023-08-02 04:50:00,079] Trial 38 finished with value: 0.9944636678200692 and parameters: {'learning_rate': 0.042803678555982425, 'num_leaves': 10, 'max_bin': 26, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [40]	training's auc: 0.996657	valid_1's auc: 0.981315
    [60]	training's auc: 0.999197	valid_1's auc: 0.991003
    [80]	training's auc: 0.999948	valid_1's auc: 0.988927
    [100]	training's auc: 1	valid_1's auc: 0.994464
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000656 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 450
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.994246	valid_1's auc: 0.979931


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:00,279] Trial 39 finished with value: 0.9951557093425606 and parameters: {'learning_rate': 0.06605645591417805, 'num_leaves': 9, 'max_bin': 18, 'feature_fraction': 0.45}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [40]	training's auc: 0.998886	valid_1's auc: 0.993772
    [60]	training's auc: 0.999922	valid_1's auc: 0.99308
    [80]	training's auc: 1	valid_1's auc: 0.994464
    [100]	training's auc: 1	valid_1's auc: 0.995156
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000592 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 725
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.997227	valid_1's auc: 0.957093


    [I 2023-08-02 04:50:00,550] Trial 40 finished with value: 0.9910034602076124 and parameters: {'learning_rate': 0.07837346902322499, 'num_leaves': 14, 'max_bin': 29, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [40]	training's auc: 0.999974	valid_1's auc: 0.978547
    [60]	training's auc: 1	valid_1's auc: 0.988927
    [80]	training's auc: 1	valid_1's auc: 0.990311
    [100]	training's auc: 1	valid_1's auc: 0.991003
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000533 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:00,765] Trial 41 finished with value: 0.9965397923875432 and parameters: {'learning_rate': 0.05237793141423932, 'num_leaves': 10, 'max_bin': 25, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [20]	training's auc: 0.991447	valid_1's auc: 0.978547
    [40]	training's auc: 0.997953	valid_1's auc: 0.985467
    [60]	training's auc: 0.999663	valid_1's auc: 0.991696
    [80]	training's auc: 1	valid_1's auc: 0.993772
    [100]	training's auc: 1	valid_1's auc: 0.99654
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000579 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 650
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:01,009] Trial 42 finished with value: 0.9930795847750865 and parameters: {'learning_rate': 0.05677274742317723, 'num_leaves': 12, 'max_bin': 26, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [20]	training's auc: 0.993521	valid_1's auc: 0.986159
    [40]	training's auc: 0.998886	valid_1's auc: 0.984083
    [60]	training's auc: 1	valid_1's auc: 0.991696
    [80]	training's auc: 1	valid_1's auc: 0.991003
    [100]	training's auc: 1	valid_1's auc: 0.99308


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:01,194] Trial 43 finished with value: 0.986159169550173 and parameters: {'learning_rate': 0.04672722182712029, 'num_leaves': 8, 'max_bin': 24, 'feature_fraction': 0.15000000000000002}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000605 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.988285	valid_1's auc: 0.946713
    [40]	training's auc: 0.994531	valid_1's auc: 0.959862
    [60]	training's auc: 0.998315	valid_1's auc: 0.970242
    [80]	training's auc: 0.999404	valid_1's auc: 0.982699
    [100]	training's auc: 0.999844	valid_1's auc: 0.986159
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000535 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 675
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [I 2023-08-02 04:50:01,440] Trial 44 finished with value: 0.9916955017301038 and parameters: {'learning_rate': 0.06350691341060608, 'num_leaves': 11, 'max_bin': 27, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [20]	training's auc: 0.995439	valid_1's auc: 0.984083
    [40]	training's auc: 0.999689	valid_1's auc: 0.984775
    [60]	training's auc: 1	valid_1's auc: 0.990311
    [80]	training's auc: 1	valid_1's auc: 0.991003
    [100]	training's auc: 1	valid_1's auc: 0.991696
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:01,659] Trial 45 finished with value: 0.9896193771626297 and parameters: {'learning_rate': 0.03466008321930528, 'num_leaves': 10, 'max_bin': 22, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.002320 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 550
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.989814	valid_1's auc: 0.977163
    [40]	training's auc: 0.995283	valid_1's auc: 0.982699
    [60]	training's auc: 0.997693	valid_1's auc: 0.988927
    [80]	training's auc: 0.999326	valid_1's auc: 0.988927
    [100]	training's auc: 0.999844	valid_1's auc: 0.989619


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000600 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.994765	valid_1's auc: 0.960554
    [40]	training's auc: 0.998497	valid_1's auc: 0.970934
    [60]	training's auc: 0.999663	valid_1's auc: 0.976471
    [80]	training's auc: 0.999611	valid_1's auc: 0.989619


    [I 2023-08-02 04:50:01,913] Trial 46 finished with value: 0.9903114186851211 and parameters: {'learning_rate': 0.027528691723569032, 'num_leaves': 13, 'max_bin': 25, 'feature_fraction': 0.15000000000000002}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:02,097] Trial 47 finished with value: 0.9889273356401385 and parameters: {'learning_rate': 0.04569787409578493, 'num_leaves': 8, 'max_bin': 24, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [100]	training's auc: 0.999819	valid_1's auc: 0.990311
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000546 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.986005	valid_1's auc: 0.959862
    [40]	training's auc: 0.99435	valid_1's auc: 0.978547
    [60]	training's auc: 0.997512	valid_1's auc: 0.983391
    [80]	training's auc: 0.999378	valid_1's auc: 0.986159
    [100]	training's auc: 0.999844	valid_1's auc: 0.988927


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:02,293] Trial 48 finished with value: 0.9868512110726644 and parameters: {'learning_rate': 0.053973537831372244, 'num_leaves': 9, 'max_bin': 21, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000531 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 525
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.990203	valid_1's auc: 0.968166
    [40]	training's auc: 0.996994	valid_1's auc: 0.974394
    [60]	training's auc: 0.999508	valid_1's auc: 0.984083
    [80]	training's auc: 0.999974	valid_1's auc: 0.986159
    [100]	training's auc: 1	valid_1's auc: 0.986851
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines


    [I 2023-08-02 04:50:02,523] Trial 49 finished with value: 0.9937716262975779 and parameters: {'learning_rate': 0.08488915277959659, 'num_leaves': 11, 'max_bin': 16, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000659 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 400
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.995879	valid_1's auc: 0.983391
    [40]	training's auc: 0.99987	valid_1's auc: 0.990311
    [60]	training's auc: 1	valid_1's auc: 0.992388
    [80]	training's auc: 1	valid_1's auc: 0.992388
    [100]	training's auc: 1	valid_1's auc: 0.993772


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000550 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 500
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.992173	valid_1's auc: 0.977855
    [40]	training's auc: 0.997486	valid_1's auc: 0.984775
    [60]	training's auc: 0.999482	valid_1's auc: 0.992388
    [80]	training's auc: 0.999922	valid_1's auc: 0.99308


    [I 2023-08-02 04:50:02,779] Trial 50 finished with value: 0.9944636678200692 and parameters: {'learning_rate': 0.037437280976222016, 'num_leaves': 13, 'max_bin': 20, 'feature_fraction': 0.45}. Best is trial 29 with value: 0.9972318339100346.


    [100]	training's auc: 1	valid_1's auc: 0.994464
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000770 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.990851	valid_1's auc: 0.975087
    [40]	training's auc: 0.997175	valid_1's auc: 0.986851
    [60]	training's auc: 0.999456	valid_1's auc: 0.992388
    [80]	training's auc: 0.999974	valid_1's auc: 0.995156


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:02,991] Trial 51 finished with value: 0.9951557093425606 and parameters: {'learning_rate': 0.04889185885999991, 'num_leaves': 10, 'max_bin': 25, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [100]	training's auc: 1	valid_1's auc: 0.995156
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000553 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 675
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.991058	valid_1's auc: 0.970934
    [40]	training's auc: 0.998575	valid_1's auc: 0.979239
    [60]	training's auc: 0.99987	valid_1's auc: 0.983391
    [80]	training's auc: 1	valid_1's auc: 0.987543


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:03,210] Trial 52 finished with value: 0.986159169550173 and parameters: {'learning_rate': 0.05453043534384284, 'num_leaves': 10, 'max_bin': 27, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:03,377] Trial 53 finished with value: 0.9820069204152249 and parameters: {'learning_rate': 0.07348910621165425, 'num_leaves': 7, 'max_bin': 22, 'feature_fraction': 0.15000000000000002}. Best is trial 29 with value: 0.9972318339100346.


    [100]	training's auc: 1	valid_1's auc: 0.986159
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000555 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 550
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.989452	valid_1's auc: 0.949481
    [40]	training's auc: 0.995594	valid_1's auc: 0.970242
    [60]	training's auc: 0.998963	valid_1's auc: 0.975087
    [80]	training's auc: 0.999715	valid_1's auc: 0.982007
    [100]	training's auc: 0.999974	valid_1's auc: 0.982007


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000743 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.991784	valid_1's auc: 0.96609
    [40]	training's auc: 0.998134	valid_1's auc: 0.979931
    [60]	training's auc: 0.999533	valid_1's auc: 0.982699
    [80]	training's auc: 0.999974	valid_1's auc: 0.984083
    [100]	training's auc: 1	valid_1's auc: 0.987543


    [I 2023-08-02 04:50:03,619] Trial 54 finished with value: 0.9875432525951557 and parameters: {'learning_rate': 0.041726376724905964, 'num_leaves': 12, 'max_bin': 24, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:03,842] Trial 55 finished with value: 0.9930795847750865 and parameters: {'learning_rate': 0.0656358540104095, 'num_leaves': 11, 'max_bin': 29, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000582 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 725
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.996035	valid_1's auc: 0.975087
    [40]	training's auc: 0.999663	valid_1's auc: 0.982007
    [60]	training's auc: 1	valid_1's auc: 0.988927
    [80]	training's auc: 1	valid_1's auc: 0.991003
    [100]	training's auc: 1	valid_1's auc: 0.99308


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:04,042] Trial 56 finished with value: 0.9910034602076124 and parameters: {'learning_rate': 0.056689912413586174, 'num_leaves': 9, 'max_bin': 23, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000608 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 575
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.987145	valid_1's auc: 0.979239
    [40]	training's auc: 0.997227	valid_1's auc: 0.987543
    [60]	training's auc: 0.999689	valid_1's auc: 0.991003
    [80]	training's auc: 0.999974	valid_1's auc: 0.991696
    [100]	training's auc: 1	valid_1's auc: 0.991003
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.002646 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 650
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [20]	training's auc: 0.998264	valid_1's auc: 0.987543
    [40]	training's auc: 1	valid_1's auc: 0.993772
    [60]	training's auc: 1	valid_1's auc: 0.995848
    [80]	training's auc: 1	valid_1's auc: 0.995156


    [I 2023-08-02 04:50:04,331] Trial 57 finished with value: 0.9951557093425606 and parameters: {'learning_rate': 0.08275663011208285, 'num_leaves': 16, 'max_bin': 26, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [100]	training's auc: 1	valid_1's auc: 0.995156
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000544 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [20]	training's auc: 0.995179	valid_1's auc: 0.986851
    [40]	training's auc: 0.999197	valid_1's auc: 0.993772
    [60]	training's auc: 0.999948	valid_1's auc: 0.995848


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:04,584] Trial 58 finished with value: 0.9965397923875432 and parameters: {'learning_rate': 0.049190367439118476, 'num_leaves': 13, 'max_bin': 25, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [80]	training's auc: 1	valid_1's auc: 0.99654
    [100]	training's auc: 1	valid_1's auc: 0.99654
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000545 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 675
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.997356	valid_1's auc: 0.983391
    [40]	training's auc: 0.999948	valid_1's auc: 0.986851


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:04,840] Trial 59 finished with value: 0.9903114186851211 and parameters: {'learning_rate': 0.06981091618230968, 'num_leaves': 13, 'max_bin': 27, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [60]	training's auc: 1	valid_1's auc: 0.991696
    [80]	training's auc: 1	valid_1's auc: 0.992388
    [100]	training's auc: 1	valid_1's auc: 0.990311
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000535 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 725
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [20]	training's auc: 0.997045	valid_1's auc: 0.977855


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:05,112] Trial 60 finished with value: 0.9923875432525952 and parameters: {'learning_rate': 0.06201842930361875, 'num_leaves': 15, 'max_bin': 29, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [40]	training's auc: 0.999819	valid_1's auc: 0.983391
    [60]	training's auc: 1	valid_1's auc: 0.989619
    [80]	training's auc: 1	valid_1's auc: 0.991003
    [100]	training's auc: 1	valid_1's auc: 0.992388
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000548 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:05,348] Trial 61 finished with value: 0.9965397923875432 and parameters: {'learning_rate': 0.04774906924683135, 'num_leaves': 12, 'max_bin': 25, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [20]	training's auc: 0.994505	valid_1's auc: 0.988235
    [40]	training's auc: 0.998626	valid_1's auc: 0.99308
    [60]	training's auc: 0.999844	valid_1's auc: 0.995848
    [80]	training's auc: 1	valid_1's auc: 0.997232
    [100]	training's auc: 1	valid_1's auc: 0.99654
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000633 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 575
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:05,615] Trial 62 finished with value: 0.9951557093425606 and parameters: {'learning_rate': 0.04762258335485442, 'num_leaves': 14, 'max_bin': 23, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [20]	training's auc: 0.994583	valid_1's auc: 0.983391
    [40]	training's auc: 0.998523	valid_1's auc: 0.99308
    [60]	training's auc: 0.999896	valid_1's auc: 0.994464
    [80]	training's auc: 1	valid_1's auc: 0.995156
    [100]	training's auc: 1	valid_1's auc: 0.995156


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:05,852] Trial 63 finished with value: 0.9910034602076124 and parameters: {'learning_rate': 0.04164166036768216, 'num_leaves': 12, 'max_bin': 13, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000581 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 325
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.990644	valid_1's auc: 0.984083
    [40]	training's auc: 0.996657	valid_1's auc: 0.986159
    [60]	training's auc: 0.999145	valid_1's auc: 0.988235
    [80]	training's auc: 0.999844	valid_1's auc: 0.989619
    [100]	training's auc: 1	valid_1's auc: 0.991003


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:06,091] Trial 64 finished with value: 0.9965397923875432 and parameters: {'learning_rate': 0.05683961741820739, 'num_leaves': 12, 'max_bin': 25, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000584 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.995879	valid_1's auc: 0.986159
    [40]	training's auc: 0.999222	valid_1's auc: 0.992388
    [60]	training's auc: 0.999948	valid_1's auc: 0.99654
    [80]	training's auc: 1	valid_1's auc: 0.995848
    [100]	training's auc: 1	valid_1's auc: 0.99654


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000543 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 650
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.994972	valid_1's auc: 0.989619
    [40]	training's auc: 0.999559	valid_1's auc: 0.995848
    [60]	training's auc: 1	valid_1's auc: 0.99654
    [80]	training's auc: 1	valid_1's auc: 0.99654


    [I 2023-08-02 04:50:06,344] Trial 65 finished with value: 0.9972318339100346 and parameters: {'learning_rate': 0.06057032788378776, 'num_leaves': 13, 'max_bin': 26, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [100]	training's auc: 1	valid_1's auc: 0.997232
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000594 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 650
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.994065	valid_1's auc: 0.985467
    [40]	training's auc: 0.998963	valid_1's auc: 0.988927
    [60]	training's auc: 1	valid_1's auc: 0.994464


    [I 2023-08-02 04:50:06,595] Trial 66 finished with value: 0.9965397923875432 and parameters: {'learning_rate': 0.0468687931698245, 'num_leaves': 13, 'max_bin': 26, 'feature_fraction': 0.45}. Best is trial 29 with value: 0.9972318339100346.


    [80]	training's auc: 1	valid_1's auc: 0.99654
    [100]	training's auc: 1	valid_1's auc: 0.99654
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000818 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 675
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.994402	valid_1's auc: 0.979931
    [40]	training's auc: 0.998108	valid_1's auc: 0.988927


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:06,856] Trial 67 finished with value: 0.9930795847750865 and parameters: {'learning_rate': 0.03453562923939209, 'num_leaves': 14, 'max_bin': 27, 'feature_fraction': 0.45}. Best is trial 29 with value: 0.9972318339100346.


    [60]	training's auc: 0.999663	valid_1's auc: 0.991696
    [80]	training's auc: 0.999974	valid_1's auc: 0.993772
    [100]	training's auc: 1	valid_1's auc: 0.99308
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000542 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 700
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.993806	valid_1's auc: 0.975779


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:07,166] Trial 68 finished with value: 0.9930795847750865 and parameters: {'learning_rate': 0.0443419646463002, 'num_leaves': 13, 'max_bin': 28, 'feature_fraction': 0.45}. Best is trial 29 with value: 0.9972318339100346.


    [40]	training's auc: 0.998626	valid_1's auc: 0.985467
    [60]	training's auc: 0.99987	valid_1's auc: 0.990311
    [80]	training's auc: 1	valid_1's auc: 0.99308
    [100]	training's auc: 1	valid_1's auc: 0.99308


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000554 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 100
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.982739	valid_1's auc: 0.957093
    [40]	training's auc: 0.993754	valid_1's auc: 0.964706
    [60]	training's auc: 0.997538	valid_1's auc: 0.96609
    [80]	training's auc: 0.999222	valid_1's auc: 0.970934


    [I 2023-08-02 04:50:07,440] Trial 69 finished with value: 0.970242214532872 and parameters: {'learning_rate': 0.05014061141606391, 'num_leaves': 15, 'max_bin': 4, 'feature_fraction': 0.45}. Best is trial 29 with value: 0.9972318339100346.


    [100]	training's auc: 0.99987	valid_1's auc: 0.970242
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000539 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 650
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [20]	training's auc: 0.996009	valid_1's auc: 0.985467
    [40]	training's auc: 0.999559	valid_1's auc: 0.988927
    [60]	training's auc: 1	valid_1's auc: 0.99308


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:07,733] Trial 70 finished with value: 0.995847750865052 and parameters: {'learning_rate': 0.057352272759048036, 'num_leaves': 16, 'max_bin': 26, 'feature_fraction': 0.45}. Best is trial 29 with value: 0.9972318339100346.


    [80]	training's auc: 1	valid_1's auc: 0.995848
    [100]	training's auc: 1	valid_1's auc: 0.995848
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000544 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.995387	valid_1's auc: 0.986851
    [40]	training's auc: 0.999715	valid_1's auc: 0.988235


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:07,988] Trial 71 finished with value: 0.9896193771626297 and parameters: {'learning_rate': 0.06092315028702674, 'num_leaves': 13, 'max_bin': 24, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [60]	training's auc: 1	valid_1's auc: 0.987543
    [80]	training's auc: 1	valid_1's auc: 0.990311
    [100]	training's auc: 1	valid_1's auc: 0.989619
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000604 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 650
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.993521	valid_1's auc: 0.988927


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:08,233] Trial 72 finished with value: 0.9965397923875432 and parameters: {'learning_rate': 0.04881155579541604, 'num_leaves': 12, 'max_bin': 26, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [40]	training's auc: 0.998834	valid_1's auc: 0.994464
    [60]	training's auc: 0.999922	valid_1's auc: 0.995156
    [80]	training's auc: 1	valid_1's auc: 0.995848
    [100]	training's auc: 1	valid_1's auc: 0.99654
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000571 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 750
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:08,500] Trial 73 finished with value: 0.9923875432525952 and parameters: {'learning_rate': 0.04665378045105613, 'num_leaves': 14, 'max_bin': 30, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [20]	training's auc: 0.995076	valid_1's auc: 0.978547
    [40]	training's auc: 0.999145	valid_1's auc: 0.985467
    [60]	training's auc: 0.999948	valid_1's auc: 0.990311
    [80]	training's auc: 1	valid_1's auc: 0.992388
    [100]	training's auc: 1	valid_1's auc: 0.992388


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000553 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 575
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.991862	valid_1's auc: 0.973702
    [40]	training's auc: 0.997097	valid_1's auc: 0.989619
    [60]	training's auc: 0.999456	valid_1's auc: 0.993772
    [80]	training's auc: 0.999974	valid_1's auc: 0.995156
    [100]	training's auc: 1	valid_1's auc: 0.995156


    [I 2023-08-02 04:50:08,742] Trial 74 finished with value: 0.9951557093425606 and parameters: {'learning_rate': 0.04131230405598405, 'num_leaves': 12, 'max_bin': 23, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000560 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 675
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.995957	valid_1's auc: 0.985467
    [40]	training's auc: 0.999456	valid_1's auc: 0.988927
    [60]	training's auc: 0.999974	valid_1's auc: 0.99308
    [80]	training's auc: 1	valid_1's auc: 0.992388
    [100]	training's auc: 1	valid_1's auc: 0.992388


    [I 2023-08-02 04:50:08,986] Trial 75 finished with value: 0.9923875432525952 and parameters: {'learning_rate': 0.05066584515972263, 'num_leaves': 13, 'max_bin': 27, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000551 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 650
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.994765	valid_1's auc: 0.984775
    [40]	training's auc: 0.999171	valid_1's auc: 0.987543
    [60]	training's auc: 1	valid_1's auc: 0.994464
    [80]	training's auc: 1	valid_1's auc: 0.994464


    [I 2023-08-02 04:50:09,233] Trial 76 finished with value: 0.9937716262975779 and parameters: {'learning_rate': 0.05453642392631183, 'num_leaves': 12, 'max_bin': 26, 'feature_fraction': 0.45}. Best is trial 29 with value: 0.9972318339100346.


    [100]	training's auc: 1	valid_1's auc: 0.993772
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000551 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 700
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.994687	valid_1's auc: 0.975087
    [40]	training's auc: 0.998834	valid_1's auc: 0.979931
    [60]	training's auc: 0.999922	valid_1's auc: 0.984775


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:09,497] Trial 77 finished with value: 0.9896193771626297 and parameters: {'learning_rate': 0.04635665076004395, 'num_leaves': 14, 'max_bin': 28, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [80]	training's auc: 1	valid_1's auc: 0.987543
    [100]	training's auc: 1	valid_1's auc: 0.989619
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000584 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 550
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.992743	valid_1's auc: 0.977855
    [40]	training's auc: 0.99689	valid_1's auc: 0.979931


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:09,739] Trial 78 finished with value: 0.9910034602076124 and parameters: {'learning_rate': 0.03645061384481447, 'num_leaves': 12, 'max_bin': 22, 'feature_fraction': 0.45}. Best is trial 29 with value: 0.9972318339100346.


    [60]	training's auc: 0.998963	valid_1's auc: 0.988927
    [80]	training's auc: 0.999819	valid_1's auc: 0.991003
    [100]	training's auc: 1	valid_1's auc: 0.991003
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000532 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.992769	valid_1's auc: 0.986159


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:09,995] Trial 79 finished with value: 0.9944636678200692 and parameters: {'learning_rate': 0.03955723565306315, 'num_leaves': 13, 'max_bin': 24, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [40]	training's auc: 0.997616	valid_1's auc: 0.99308
    [60]	training's auc: 0.999585	valid_1's auc: 0.99308
    [80]	training's auc: 0.999948	valid_1's auc: 0.994464
    [100]	training's auc: 1	valid_1's auc: 0.994464
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000522 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:10,230] Trial 80 finished with value: 0.9972318339100346 and parameters: {'learning_rate': 0.060150517879146465, 'num_leaves': 11, 'max_bin': 25, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [20]	training's auc: 0.993806	valid_1's auc: 0.980623
    [40]	training's auc: 0.999093	valid_1's auc: 0.987543
    [60]	training's auc: 0.999974	valid_1's auc: 0.995848
    [80]	training's auc: 1	valid_1's auc: 0.99654
    [100]	training's auc: 1	valid_1's auc: 0.997232
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000651 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [I 2023-08-02 04:50:10,456] Trial 81 finished with value: 0.9951557093425606 and parameters: {'learning_rate': 0.06412023198348192, 'num_leaves': 11, 'max_bin': 25, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [20]	training's auc: 0.993961	valid_1's auc: 0.981315
    [40]	training's auc: 0.999482	valid_1's auc: 0.987543
    [60]	training's auc: 1	valid_1's auc: 0.99308
    [80]	training's auc: 1	valid_1's auc: 0.994464
    [100]	training's auc: 1	valid_1's auc: 0.995156
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000547 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 650
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:10,687] Trial 82 finished with value: 0.9916955017301038 and parameters: {'learning_rate': 0.05746847738110458, 'num_leaves': 11, 'max_bin': 26, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [20]	training's auc: 0.992614	valid_1's auc: 0.984775
    [40]	training's auc: 0.998808	valid_1's auc: 0.986159
    [60]	training's auc: 0.999948	valid_1's auc: 0.99308
    [80]	training's auc: 1	valid_1's auc: 0.991696
    [100]	training's auc: 1	valid_1's auc: 0.991696
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000542 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:10,927] Trial 83 finished with value: 0.9896193771626297 and parameters: {'learning_rate': 0.04405283689760378, 'num_leaves': 12, 'max_bin': 24, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [20]	training's auc: 0.992536	valid_1's auc: 0.971626
    [40]	training's auc: 0.998497	valid_1's auc: 0.982007
    [60]	training's auc: 0.999585	valid_1's auc: 0.984775
    [80]	training's auc: 0.999974	valid_1's auc: 0.985467
    [100]	training's auc: 1	valid_1's auc: 0.989619
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000535 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 525
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.99378	valid_1's auc: 0.977163
    [40]	training's auc: 0.998911	valid_1's auc: 0.983391
    [60]	training's auc: 0.999948	valid_1's auc: 0.988927
    [80]	training's auc: 1	valid_1's auc: 0.991696
    [100]	training's auc: 1	valid_1's auc: 0.990311


    [I 2023-08-02 04:50:11,174] Trial 84 finished with value: 0.9903114186851211 and parameters: {'learning_rate': 0.049356941576165485, 'num_leaves': 12, 'max_bin': 21, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000651 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 675
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.995724	valid_1's auc: 0.984083
    [40]	training's auc: 0.99943	valid_1's auc: 0.990311
    [60]	training's auc: 0.999974	valid_1's auc: 0.99308
    [80]	training's auc: 1	valid_1's auc: 0.990311


    [I 2023-08-02 04:50:11,428] Trial 85 finished with value: 0.9910034602076124 and parameters: {'learning_rate': 0.05226911928411247, 'num_leaves': 13, 'max_bin': 27, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [100]	training's auc: 1	valid_1's auc: 0.991003
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000608 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.993236	valid_1's auc: 0.981315
    [40]	training's auc: 0.998886	valid_1's auc: 0.989619
    [60]	training's auc: 1	valid_1's auc: 0.995848
    [80]	training's auc: 1	valid_1's auc: 0.995848


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:11,656] Trial 86 finished with value: 0.9965397923875432 and parameters: {'learning_rate': 0.05789776555132105, 'num_leaves': 11, 'max_bin': 25, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [100]	training's auc: 1	valid_1's auc: 0.99654
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000520 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 725
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.973253	valid_1's auc: 0.857439
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [40]	training's auc: 0.98686	valid_1's auc: 0.868512
    [60]	training's auc: 0.99168	valid_1's auc: 0.90173
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [80]	training's auc: 0.994454	valid_1's auc: 0.90519


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:11,868] Trial 87 finished with value: 0.9224913494809689 and parameters: {'learning_rate': 0.06786911304331161, 'num_leaves': 10, 'max_bin': 29, 'feature_fraction': 0.05}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [100]	training's auc: 0.996942	valid_1's auc: 0.922491
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000559 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [20]	training's auc: 0.997123	valid_1's auc: 0.986851
    [40]	training's auc: 0.999819	valid_1's auc: 0.990311
    [60]	training's auc: 1	valid_1's auc: 0.993772


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:12,135] Trial 88 finished with value: 0.9923875432525952 and parameters: {'learning_rate': 0.06063856304834263, 'num_leaves': 14, 'max_bin': 25, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [80]	training's auc: 1	valid_1's auc: 0.993772
    [100]	training's auc: 1	valid_1's auc: 0.992388
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000513 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 575
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.992847	valid_1's auc: 0.955017
    [40]	training's auc: 0.998004	valid_1's auc: 0.97301


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:12,387] Trial 89 finished with value: 0.9889273356401385 and parameters: {'learning_rate': 0.043708193668384, 'num_leaves': 13, 'max_bin': 23, 'feature_fraction': 0.15000000000000002}. Best is trial 29 with value: 0.9972318339100346.


    [60]	training's auc: 0.999611	valid_1's auc: 0.97301
    [80]	training's auc: 0.999922	valid_1's auc: 0.986159
    [100]	training's auc: 1	valid_1's auc: 0.988927
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000654 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 700
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [20]	training's auc: 0.996294	valid_1's auc: 0.977855


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:12,673] Trial 90 finished with value: 0.9923875432525952 and parameters: {'learning_rate': 0.05340212522019517, 'num_leaves': 15, 'max_bin': 28, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [40]	training's auc: 0.999456	valid_1's auc: 0.986159
    [60]	training's auc: 1	valid_1's auc: 0.992388
    [80]	training's auc: 1	valid_1's auc: 0.991003
    [100]	training's auc: 1	valid_1's auc: 0.992388
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000273 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 650
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:12,963] Trial 91 finished with value: 0.9910034602076124 and parameters: {'learning_rate': 0.058114694197598855, 'num_leaves': 11, 'max_bin': 26, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [20]	training's auc: 0.992899	valid_1's auc: 0.985467
    [40]	training's auc: 0.998626	valid_1's auc: 0.988927
    [60]	training's auc: 0.999948	valid_1's auc: 0.992388
    [80]	training's auc: 1	valid_1's auc: 0.992388
    [100]	training's auc: 1	valid_1's auc: 0.991003


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:13,195] Trial 92 finished with value: 0.995847750865052 and parameters: {'learning_rate': 0.04859410837121232, 'num_leaves': 11, 'max_bin': 25, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000564 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.991732	valid_1's auc: 0.977855
    [40]	training's auc: 0.997616	valid_1's auc: 0.986159
    [60]	training's auc: 0.999585	valid_1's auc: 0.993772
    [80]	training's auc: 0.999974	valid_1's auc: 0.993772
    [100]	training's auc: 1	valid_1's auc: 0.995848


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000584 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 650
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.994505	valid_1's auc: 0.987543
    [40]	training's auc: 0.999378	valid_1's auc: 0.988927
    [60]	training's auc: 1	valid_1's auc: 0.993772
    [80]	training's auc: 1	valid_1's auc: 0.991696
    [100]	training's auc: 1	valid_1's auc: 0.993772


    [I 2023-08-02 04:50:13,438] Trial 93 finished with value: 0.9937716262975779 and parameters: {'learning_rate': 0.06462954399795567, 'num_leaves': 12, 'max_bin': 26, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:13,654] Trial 94 finished with value: 0.9910034602076124 and parameters: {'learning_rate': 0.0716964245949572, 'num_leaves': 10, 'max_bin': 24, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000556 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.992743	valid_1's auc: 0.964706
    [40]	training's auc: 0.999248	valid_1's auc: 0.982007
    [60]	training's auc: 1	valid_1's auc: 0.988235
    [80]	training's auc: 1	valid_1's auc: 0.988927
    [100]	training's auc: 1	valid_1's auc: 0.991003


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:13,883] Trial 95 finished with value: 0.9937716262975779 and parameters: {'learning_rate': 0.038979449053546014, 'num_leaves': 11, 'max_bin': 27, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000543 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 675
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.992743	valid_1's auc: 0.984083
    [40]	training's auc: 0.997097	valid_1's auc: 0.990311
    [60]	training's auc: 0.999197	valid_1's auc: 0.992388
    [80]	training's auc: 0.99987	valid_1's auc: 0.993772
    [100]	training's auc: 1	valid_1's auc: 0.993772


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000516 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 550
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.992354	valid_1's auc: 0.976471
    [40]	training's auc: 0.999119	valid_1's auc: 0.989619
    [60]	training's auc: 0.999948	valid_1's auc: 0.994464
    [80]	training's auc: 1	valid_1's auc: 0.99308
    [100]	training's auc: 1	valid_1's auc: 0.993772


    [I 2023-08-02 04:50:14,126] Trial 96 finished with value: 0.9937716262975779 and parameters: {'learning_rate': 0.05426785236961393, 'num_leaves': 12, 'max_bin': 22, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:14,337] Trial 97 finished with value: 0.9965397923875432 and parameters: {'learning_rate': 0.05994602228311251, 'num_leaves': 9, 'max_bin': 25, 'feature_fraction': 0.25}. Best is trial 29 with value: 0.9972318339100346.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000548 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 625
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.989633	valid_1's auc: 0.975779
    [40]	training's auc: 0.99816	valid_1's auc: 0.988927
    [60]	training's auc: 0.999715	valid_1's auc: 0.995156
    [80]	training's auc: 1	valid_1's auc: 0.995848
    [100]	training's auc: 1	valid_1's auc: 0.99654


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.


    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000528 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 700
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.997486	valid_1's auc: 0.92872
    [40]	training's auc: 0.999689	valid_1's auc: 0.960554
    [60]	training's auc: 1	valid_1's auc: 0.959862
    [80]	training's auc: 1	valid_1's auc: 0.977163


    [I 2023-08-02 04:50:14,952] Trial 98 finished with value: 0.9785467128027682 and parameters: {'learning_rate': 0.0506478792148721, 'num_leaves': 13, 'max_bin': 28, 'feature_fraction': 0.15000000000000002}. Best is trial 29 with value: 0.9972318339100346.


    [100]	training's auc: 1	valid_1's auc: 0.978547
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000611 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.991888	valid_1's auc: 0.977855
    [40]	training's auc: 0.996968	valid_1's auc: 0.987543
    [60]	training's auc: 0.999093	valid_1's auc: 0.990311
    [80]	training's auc: 0.999896	valid_1's auc: 0.989619


    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [LightGBM] [Fatal] Cannot change max_bin after constructed Dataset handle.
    [I 2023-08-02 04:50:15,172] Trial 99 finished with value: 0.9889273356401385 and parameters: {'learning_rate': 0.0453640560972869, 'num_leaves': 10, 'max_bin': 24, 'feature_fraction': 0.35000000000000003}. Best is trial 29 with value: 0.9972318339100346.


    [100]	training's auc: 1	valid_1's auc: 0.988927



```python
model_params = {
    'boosting': 'gbdt', # dart 提升算法
    'objective': 'binary', # 均方误差
    'metric': 'auc', # # 均方根误差
    'learning_rate':  0.0453640560972869,  # 学习率
    'seed': 42, # 随机种子,
     'num_leaves': 10, # 最大叶子数量
    'max_bin': 24, # 最大分箱数
    'feature_fraction': 0.35, # 特征采样比例
}
```


```python
seeds = [3,5,111,129,223]

for seed in seeds:
    _model_params = dict(model_params)
    _model_params["seed"] = seed # 随机种子

    log_callback = lgb.log_evaluation(period=20) # 训练日志频率

    model = lgb.train(
        params=_model_params, # 参数
        train_set=train_dset, # 训练集
        valid_sets=[train_dset], # 验证集
        callbacks=[log_callback,], # 训练日志
    )

    model.save_model(f"lgbm_seed{seed}.txt")
```

    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001069 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.99067
    [40]	training's auc: 0.997227
    [60]	training's auc: 0.999482
    [80]	training's auc: 1
    [100]	training's auc: 1
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000589 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.991395
    [40]	training's auc: 0.997123
    [60]	training's auc: 0.999508
    [80]	training's auc: 0.999948
    [100]	training's auc: 1
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000690 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.99194
    [40]	training's auc: 0.996864
    [60]	training's auc: 0.99943
    [80]	training's auc: 0.999974
    [100]	training's auc: 1
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000548 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.992199
    [40]	training's auc: 0.997927
    [60]	training's auc: 0.999352
    [80]	training's auc: 0.999896
    [100]	training's auc: 1
    [LightGBM] [Info] Number of positive: 91, number of negative: 424
    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000534 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 600
    [LightGBM] [Info] Number of data points in the train set: 515, number of used features: 25
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.176699 -> initscore=-1.538874
    [LightGBM] [Info] Start training from score -1.538874
    [20]	training's auc: 0.992899
    [40]	training's auc: 0.998056
    [60]	training's auc: 0.999352
    [80]	training's auc: 0.999922
    [100]	training's auc: 1



```python
    lgb.plot_importance(model, figsize=(8,15), importance_type="split", max_num_features=30) # split 特征重要度
    lgb.plot_importance(model, figsize=(8,15), importance_type="gain", max_num_features=30) # gain 特征重要度

    plt.show()
```



![png](output_15_0.png)





![png](output_15_1.png)




```python
files = glob("./lgbm_seed*.txt")
display(files)

boosters_lgbm_linear_dart = [lgb.Booster(model_file=fn) for fn in files] # 导入训练好的模型
display(boosters_lgbm_linear_dart)
```


    ['./lgbm_seed3.txt',
     './lgbm_seed129.txt',
     './lgbm_seed5.txt',
     './lgbm_seed111.txt',
     './lgbm_seed223.txt']



    [<lightgbm.basic.Booster at 0x7b5ff396b520>,
     <lightgbm.basic.Booster at 0x7b6003218130>,
     <lightgbm.basic.Booster at 0x7b5ff2cbfdc0>,
     <lightgbm.basic.Booster at 0x7b600bf8a320>,
     <lightgbm.basic.Booster at 0x7b5ff39e2cb0>]



```python
def predict(boosters, dataframe ):
    features = feat_names
    for model in boosters:
        pos =  [model.predict(
            dataframe[features], # 特征数据
            start_iteration=0, # 起始迭代次数
            num_iteration=model.current_iteration(), # 迭代次数
        )]
        nag = [1-p for p in pos]


    return np.mean(pos, axis=0),np.mean(nag, axis=0) # 各个模型预测结果的均值
```


```python
test_data = import_data('/kaggle/input/icr-identify-age-related-conditions/test.csv')
```

    Mem. usage decreased to  0.00 Mb (35.0% reduction)



```python
test_data['cat_EJ'] = test_data['EJ'].cat.codes
```


```python
pos,nag = predict(boosters_lgbm_linear_dart, test_data)
```


```python
submission_df = test_data[['Id']]
submission_df['class_0'] = nag
submission_df['class_1'] = pos
```

    /tmp/ipykernel_32/3100712703.py:2: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      submission_df['class_0'] = nag
    /tmp/ipykernel_32/3100712703.py:3: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      submission_df['class_1'] = pos



```python
submission_df.to_csv('submission.csv', index = False)
```


```python

```
