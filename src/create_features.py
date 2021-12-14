import datetime
import time
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, PSARIndicator, KSTIndicator, MassIndex

# メモリ使用量削減用
def reduce_mem_usage(df, verbose=True):
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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# 各処理にかかった時間を計測
class Timer(object):
    def __init__(self, message=''):
        self.message = message
        self.timezone = datetime.timezone(datetime.timedelta(hours=+9), 'JST')

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(f'{datetime.datetime.now(self.timezone):%Y-%m-%d %H:%M:%S}',
              self.message, f'実行時間: {(time.time()-self.start)/60:.2f}分')

# ラグ特徴量
def create_lags(group_df, val_col, lags, prefix):
    lag_df = pd.DataFrame()

    for lag in lags:
        lag_df[f'{prefix}lag_{lag}'] = group_df[val_col].transform(
            lambda x: x.shift(lag))

    return lag_df

# 単純移動平均、標準偏差
def create_rolls(group_df, val_col, lags, rolls, prefix):
    roll_df = pd.DataFrame()

    for lag in lags:
        for roll in rolls:
            roll_df[f'{prefix}rmean_{lag}_{roll}'] = group_df[val_col].transform(
                lambda x: x.shift(lag).rolling(roll).mean())
            roll_df[f'{prefix}rstd_{lag}_{roll}'] = group_df[val_col].transform(
                lambda x: x.shift(lag).rolling(roll).std())

    return roll_df

# 指数移動平均乖離率
def create_ema_gap(df, group_df, group_col, val_col, rolls, prefix):
    ema_df = pd.DataFrame(df[group_col]) if group_col else pd.DataFrame()
    for i in rolls:
        ema_df[f'{prefix}EMA_{i}'] = group_df[val_col].transform(
            lambda x: x.shift(1).ewm(span=i).mean())
        ema_df[f'{prefix}EMA_GAP_{i}'] = df[val_col].shift(
            1) / ema_df[f'{prefix}EMA_{i}'] - 1
        ema_df[f'{prefix}EMA_GAP_{i}'].fillna(0, inplace=True)

        ema_df.drop(f'{prefix}EMA_{i}', axis=1, inplace=True)

    if group_col:
        ema_df.drop(group_col, axis=1, inplace=True)
    ema_df.sort_index(axis=1, inplace=True)

    return ema_df

# リターン
def create_return(group_df, val_col, rolls, prefix):
    return_df = pd.DataFrame()

    for i in rolls:
        return_df[f'{prefix}Return_{i}'] = group_df[val_col].pct_change(i)
        return_df[f'{prefix}Return_{i}'].fillna(0, inplace=True)

    return return_df

# ヒストリカルボラティリティ
def create_hv(df, group_df, group_col, val_col, rolls, prefix):
    hv_df = pd.DataFrame(df[group_col]) if group_col else pd.DataFrame()
    hv_df['Variability'] = group_df[val_col].pct_change(1)
    hv_group_df = hv_df.groupby(group_col) if group_col else hv_df.copy()

    for i in rolls:
        hv_df[f'{prefix}HV_{i}'] = hv_group_df['Variability'].rolling(
            i).std().reset_index(drop=True)
        hv_df[f'{prefix}HV_{i}'].fillna(0, inplace=True)

    hv_df.drop('Variability', axis=1, inplace=True)
    if group_col:
        hv_df.drop(group_col, axis=1, inplace=True)

    return hv_df

# Skew
def create_skew(df, group_df, group_col, val_col, rolls, prefix):
    skew_df = pd.DataFrame(df[group_col]) if group_col else pd.DataFrame()

    for i in rolls:
        skew_df[f'{prefix}Skew_{i}'] = group_df[val_col].rolling(
            window=i).skew().reset_index(drop=True)
        skew_df[f'{prefix}Skew_{i}'].fillna(0, inplace=True)

    if group_col:
        skew_df.drop(group_col, axis=1, inplace=True)
    skew_df.sort_index(axis=1, inplace=True)

    return skew_df

# Kurt
def create_kurt(group_df, val_col, rolls, prefix):
    kurt_df = pd.DataFrame()

    for i in rolls:
        kurt_df[f'{prefix}Kurt_{i}'] = group_df[val_col].rolling(
            window=i).kurt().reset_index(drop=True)
        kurt_df[f'{prefix}Kurt_{i}'].fillna(0, inplace=True)

    return kurt_df

# 分散
def create_var(group_df, val_col, rolls, prefix):
    var_df = pd.DataFrame()

    for i in rolls:
        var_df[f'{prefix}Var_{i}'] = group_df[val_col].rolling(
            window=i).var().reset_index(drop=True)
        var_df[f'{prefix}Var_{i}'].fillna(0, inplace=True)

    return var_df

# 最小値、最大値
def create_min_max(group_df, val_col, rolls, prefix):
    min_max_df = pd.DataFrame()

    for i in rolls:
        min_max_df[f'{prefix}MIN_{i}'] = group_df[val_col].rolling(
            window=i).min().reset_index(drop=True)
        min_max_df[f'{prefix}MIN_{i}'].fillna(0, inplace=True)
        min_max_df[f'{prefix}MAX_{i}'] = group_df[val_col].rolling(
            window=i).max().reset_index(drop=True)
        min_max_df[f'{prefix}MAX_{i}'].fillna(0, inplace=True)

    return min_max_df

# 前週から価格が上昇したか、下落したか
def create_up_down(df, group_col, val_col, rolls, prefix):
    up_down_df = pd.DataFrame(
        df[group_col]) if group_col else pd.DataFrame(index=df.index)

    up_down_df[f'{prefix}Up'] = 0
    up_down_df[f'{prefix}Down'] = 0
    up_down_df.loc[df[val_col] > 0, f'{prefix}Up'] = 1
    up_down_df.loc[df[val_col] < 0, f'{prefix}Down'] = 1
    up_down_group_df = up_down_df.groupby(
        group_col) if group_col else up_down_df.copy()
    up_down_df[f'{prefix}Total_Up'] = up_down_group_df[f'{prefix}Up'].cumsum()
    up_down_df[f'{prefix}Total_Down'] = up_down_group_df[f'{prefix}Down'].cumsum()
    up_down_group_df = up_down_df.groupby(
        group_col) if group_col else up_down_df.copy()

    for i in rolls:
        up_down_df[f'{prefix}Up_{i}'] = (up_down_group_df[f'{prefix}Total_Up'].shift(
            0) - up_down_group_df[f'{prefix}Total_Up'].shift(i)) / i
        up_down_df[f'{prefix}Up_{i}'].fillna(method='bfill', inplace=True)

    for i in rolls:
        up_down_df[f'{prefix}Down_{i}'] = (up_down_group_df[f'{prefix}Total_Down'].shift(
            0) - up_down_group_df[f'{prefix}Total_Down'].shift(i)) / i
        up_down_df[f'{prefix}Down_{i}'].fillna(method='bfill', inplace=True)

    drop_columns = ['id', f'{prefix}Up', f'{prefix}Down', f'{prefix}Total_Up',
                    f'{prefix}Total_Down', f'{prefix}Up_str', f'{prefix}Down_str']
    drop_columns += [f'{prefix}Up_str_{i}' for i in [13, 26]]
    drop_columns += [f'{prefix}Down_str_{i}' for i in [13, 26]]

    return up_down_df.drop(drop_columns, axis=1, errors='ignore')

# AverageTrueRange
def create_atr(group_df, val_col, prefix):
    atr_df = pd.DataFrame()

    if type(group_df) == pd.DataFrame:
        group_df = {'All': group_df}.items()

    for key, value in group_df:
        indicator_atr = AverageTrueRange(
            high=value[val_col], low=value[val_col], close=value[val_col], window=9, fillna=True)
        tmp_atr_df = indicator_atr.average_true_range()
        tmp_atr_df = pd.concat([tmp_atr_df, tmp_atr_df.pct_change(1)], axis=1)
        atr_df = pd.concat([atr_df, tmp_atr_df])
    atr_df.columns = [f'{prefix}ATR', f'{prefix}ATR_Change']
    atr_df[f'{prefix}ATR_Change'].fillna(0, inplace=True)
    atr_df.loc[atr_df[f'{prefix}ATR_Change'].abs() == float(
        'inf'), f'{prefix}ATR_Change'] = 0

    return atr_df

# RSIIndicator
def create_rsi(group_df, val_col, prefix):
    rsi_df = pd.DataFrame()

    if type(group_df) == pd.DataFrame:
        group_df = {'All': group_df}.items()

    for key, value in group_df:
        indicator_rsi = RSIIndicator(
            close=value[val_col], window=9, fillna=False)
        tmp_rsi_df = indicator_rsi.rsi()

        # RSI 0 を0.5に置換
        tmp_rsi_df.loc[tmp_rsi_df == 0] = 0.5

        tmp_rsi_df = pd.concat([tmp_rsi_df, tmp_rsi_df.pct_change(1)], axis=1)
        rsi_df = pd.concat([rsi_df, tmp_rsi_df])
    rsi_df.columns = [f'{prefix}RSI', f'{prefix}RSI_Change']
    rsi_df[f'{prefix}RSI'].fillna(50, inplace=True)
    rsi_df[f'{prefix}RSI_Change'].fillna(0, inplace=True)

    return rsi_df

# MACD
def create_macd(group_df, val_col, prefix):
    macd_df = pd.DataFrame()

    if type(group_df) == pd.DataFrame:
        group_df = {'All': group_df}.items()

    for key, value in group_df:
        indicator_macd = MACD(
            close=value[val_col], window_slow=12, window_fast=9, fillna=True)
        tmp_macd_df = pd.DataFrame()
        tmp_macd_df[f'{prefix}MACD'] = indicator_macd.macd()
        tmp_macd_df[f'{prefix}MACD_Diff'] = indicator_macd.macd_diff()
        tmp_macd_df[f'{prefix}MACD_Signal'] = indicator_macd.macd_signal()
        macd_df = pd.concat([macd_df, tmp_macd_df])

    return macd_df

# BollingerBands
def create_bb(group_df, val_col, prefix):
    bb_df = pd.DataFrame()

    if type(group_df) == pd.DataFrame:
        group_df = {'All': group_df}.items()

    for key, value in group_df:
        indicator_macd = BollingerBands(close=value[val_col], fillna=True)
        tmp_bb_df = indicator_macd.bollinger_pband()
        tmp_bb_df = pd.concat([tmp_bb_df, tmp_bb_df.pct_change(1)], axis=1)
        bb_df = pd.concat([bb_df, tmp_bb_df])
    bb_df.columns = [f'{prefix}BBP', f'{prefix}BBP_Change']
    bb_df[f'{prefix}BBP_Change'].fillna(0, inplace=True)
    bb_df.loc[bb_df[f'{prefix}BBP_Change'].abs() == float(
        'inf'), f'{prefix}BBP_Change'] = 0

    return bb_df

# StochasticOscillator
def create_stock(group_df, val_col, prefix):
    stock_df = pd.DataFrame()

    if type(group_df) == pd.DataFrame:
        group_df = {'All': group_df}.items()

    for key, value in group_df:
        indicator_stoch = StochasticOscillator(
            close=value[val_col], high=value[val_col], low=value[val_col], fillna=True)
        tmp_stock_df = pd.DataFrame()
        tmp_stock_df[f'{prefix}Stoch'] = indicator_stoch.stoch()
        tmp_stock_df[f'{prefix}Stoch_Signal'] = indicator_stoch.stoch_signal()
        stock_df = pd.concat([stock_df, tmp_stock_df])

    return stock_df

# PSARIndicator
def create_parabolic(group_df, val_col, prefix):
    parabolic_df = pd.DataFrame()

    if type(group_df) == pd.DataFrame:
        group_df = {'All': group_df}.items()

    for key, value in group_df:
        indicator_parabolic = PSARIndicator(
            close=value[val_col], high=value[val_col], low=value[val_col], fillna=True)
        tmp_parabolic_df = pd.DataFrame()
        tmp_parabolic_df[f'{prefix}PSAR_Down'] = indicator_parabolic.psar_down()
        tmp_parabolic_df[f'{prefix}PSAR_Down_Ind'] = indicator_parabolic.psar_down_indicator(
        )
        tmp_parabolic_df[f'{prefix}PSAR_Up'] = indicator_parabolic.psar_up()
        tmp_parabolic_df[f'{prefix}PSAR_Up_Ind'] = indicator_parabolic.psar_up_indicator(
        )
        parabolic_df = pd.concat([parabolic_df, tmp_parabolic_df])

    return parabolic_df

# KSTIndicator
def create_kst(group_df, val_col, prefix):
    kst_df = pd.DataFrame()

    if type(group_df) == pd.DataFrame:
        group_df = {'All': group_df}.items()

    for key, value in group_df:
        indicator_kst = KSTIndicator(close=value[val_col], fillna=True)
        tmp_kst_df = pd.DataFrame()
        tmp_kst_df[f'{prefix}KST'] = indicator_kst.kst()
        tmp_kst_df[f'{prefix}KST_Diff'] = indicator_kst.kst_diff()
        tmp_kst_df[f'{prefix}KST_Sig'] = indicator_kst.kst_sig()
        kst_df = pd.concat([kst_df, tmp_kst_df])

    return kst_df

# シャープレシオ
def sharp_ratio(group_df1, group_df2, val_col, rolls):
    sharp_df = pd.DataFrame()

    ticker_df = create_return(group_df1, val_col, [1], '')
    market_df = create_return(group_df2, f'Market_{val_col}', [1], 'Market_')
    market_df = pd.DataFrame(market_df.values.tolist()
                             * (len(ticker_df) // len(market_df)))
    ticker_df = ticker_df.join(market_df)
    ticker_df.columns = ['t_value', 'm_value']

    for i in rolls:
        sharp_df[f'Sharpe_Ratio_{i}'] = ((ticker_df.t_value.rolling(i, min_periods=1).mean() - ticker_df.m_value.rolling(i, min_periods=1).mean()) /
                                         ticker_df.t_value.rolling(i, min_periods=1).std().fillna(method='bfill'))
        sharp_df.loc[sharp_df[f'Sharpe_Ratio_{i}'].abs() == float(
            'inf'), f'Sharpe_Ratio_{i}'] = np.nan
        sharp_df[f'Sharpe_Ratio_{i}'].fillna(method='bfill', inplace=True)

    return sharp_df

# 標準化
def standardization(df):
    drop_columns = ['Date', 'id', 'y', 'y_prev',
                    'y_diff', 'y_diff_std', 'y_diff_norm', 'List']
    res = df.drop(drop_columns, axis=1)

    res = (res - res.mean()) / res.std()
    res.dropna(how='any', axis=1, inplace=True)
    res.drop(res.columns[(res.abs() == float('inf')).sum()
             > 0], axis=1, inplace=True)

    return res

# PCA
def pca(df, n=2):
    def make_radian_row(pca_result):
        rad = []
        for r in pca_result:
            rad.append(math.atan(r[0]/r[1]))
        return rad

    scaled_df = standardization(df)
    res = pd.DataFrame(index=scaled_df.index)

    pca = PCA(n_components=n)
    trans = pca.fit_transform(scaled_df)
    res['PCA'] = make_radian_row(trans)
    res = res.join(pd.DataFrame(
        {'PCA1': trans.T[0], 'PCA2': trans.T[1]}, index=scaled_df.index))

    return res

# K-Means
def k_means(df, n=3):
    scaled_df = standardization(df)

    cls = KMeans(n_clusters=n)
    pred = cls.fit_predict(scaled_df)

    res = pd.DataFrame(index=scaled_df.index)
    columns = scaled_df.columns
    res['d1'] = np.sqrt(
        ((scaled_df[columns] - cls.cluster_centers_[0]) ** 2).sum(axis=1))
    res['d2'] = np.sqrt(
        ((scaled_df[columns] - cls.cluster_centers_[1]) ** 2).sum(axis=1))
    res['d3'] = np.sqrt(
        ((scaled_df[columns] - cls.cluster_centers_[2]) ** 2).sum(axis=1))
    res['d'] = res[['d1', 'd2', 'd3']].min(axis=1)
    res['ClusterId'] = pred
    res.drop(['d1', 'd2', 'd3'], axis=1, inplace=True)

    return res
