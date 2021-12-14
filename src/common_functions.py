import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split, KFold

class Config:
    submission_date = '2019-11-24'
    input_dir_name = './input'
    initial_dir_name = './output/initial_data'
    intermediate_dir_name = './output/intermediate_submission'
    submission_dir_name = './output/submission'

# pandasの表示設定
def set_pandas_options():
    pd.set_option('display.max_columns', 300)
    pd.set_option('display.max_rows', 150)

# 乱数固定
def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# CreateInitialData.ipynbで作成したデータを読み込み
def read_initial_data():
    price_df2 = pd.read_csv(f'{Config.initial_dir_name}/train_data2.csv.gzip', index_col=0, parse_dates=True, compression='gzip',
                            dtype={'y': object, 'y_prev': object, 'y_diff': object, 'y_diff_std': object, 'y_diff_norm': object, 'IPOyear': object})
    price_df2.Date = price_df2.Date.astype(np.datetime64)
    for c in ['y', 'y_prev', 'y_diff', 'y_diff_std', 'y_diff_norm', 'IPOyear']:
        price_df2[c] = price_df2[c].apply(float)
    company_df2 = pd.read_csv(f'{Config.initial_dir_name}/company_list2.csv.gzip', index_col=0, compression='gzip')
    submission_df = pd.read_csv(f'{Config.input_dir_name}/submission_template.csv')

    return price_df2, company_df2, submission_df
