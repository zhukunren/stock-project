# app.py

import sys
import os
import numpy as np
import pandas as pd
import math
from datetime import datetime
import multiprocessing
from itertools import combinations
from math import comb
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, precision_score, 
    recall_score, average_precision_score, matthews_corrcoef, f1_score
)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

# ============== PyTorch & Skorch 相关 ==============
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

# ============== Plotly 相关 ==============
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ============== 从外部 function.py 导入公用函数和指标函数 ==============
from function import *

# ============== 引入 Streamlit 进行Web展示 ==============
import streamlit as st
from joblib import Parallel, delayed

# ============== 导入自定义模型类 ==============
from models import (
    WeightedBCEWithLogitsLoss,
    WeightedCrossEntropyLoss,
    TransformerClassifier,
    MLPClassifierModule
)

# ============== AgGrid 相关 ==============
from st_aggrid import AgGrid

# ============== 其他必要导入 ==============
import base64
import time

# ============== 导入 Tushare 并设置 Token ==============
import tushare as ts

# 设置您的 Tushare Token
ts.set_token('c5c5700a6f4678a1837ad234f2e9ea2a573a26b914b47fa2dbb38aff')
pro = ts.pro_api()

##############################################################################
#                            设置随机种子
##############################################################################
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 对于某些操作，可能需要设置以下参数以确保完全的确定性
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(42)

# ---------- 数据读取与处理函数 ----------

def read_day_from_tushare(symbol_code, symbol_type='stock'):
    """
    使用 Tushare API 获取股票或指数的全部日线行情数据。
    参数:
    - symbol_code: 股票或指数代码 (如 "000001.SZ" 或 "000300.SH")
    - symbol_type: 'stock' 或 'index' (不区分大小写)
    返回:
    - 包含日期、开高低收、成交量等列的DataFrame
    """
    symbol_type = symbol_type.lower()
    print(f"传递给 read_day_from_tushare 的 symbol_type: {symbol_type} (类型: {type(symbol_type)})")  # 调试输出
    print(f"尝试通过 Tushare 获取{symbol_type}数据: {symbol_code}")
    
    # 添加断言，确保 symbol_type 是 'stock' 或 'index'
    assert symbol_type in ['stock', 'index'], "symbol_type 必须是 'stock' 或 'index'"
    
    try:
        if symbol_type == 'stock':
            # 获取股票日线数据
            df = pro.daily(ts_code=symbol_code, start_date='20000101', end_date='20251231')
            if df.empty:
                print("Tushare 返回的股票数据为空。")
                return pd.DataFrame()
            
            # 转换日期格式并排序
            df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('date')
            
            # 重命名和选择需要的列
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'vol': 'Volume',
                'amount': 'Amount',
                'trade_date': 'TradeDate'
            })
            df.set_index('date', inplace=True)
            
            # 选择需要的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'TradeDate']
            available_columns = [col for col in required_columns if col in df.columns]
            df = df[available_columns]
        
        elif symbol_type == 'index':
            # 获取指数日线数据，使用 index_daily 接口
            df = pro.index_daily(ts_code=symbol_code, start_date='20000101', end_date='20251231')
            if df.empty:
                print("Tushare 返回的指数数据为空。")
                return pd.DataFrame()
            
            # 转换日期格式并排序
            df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('date')
            
            # 重命名和选择需要的列
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'vol': 'Volume',
                'amount': 'Amount',
                'trade_date': 'TradeDate'
            })
            df.set_index('date', inplace=True)
            
            # 选择需要的列，处理可能缺失的字段
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'TradeDate']
            available_columns = [col for col in required_columns if col in df.columns]
            df = df[available_columns]
        
        print(f"通过 Tushare 获取了 {len(df)} 条记录。")
        print(f"数据框的列：{df.columns.tolist()}")
        print(f"数据框前5行：\n{df.head()}")
        return df
    except AssertionError as ae:
        print(f"断言错误：{ae}")
        return pd.DataFrame()
    except Exception as e:
        print(f"通过 Tushare 获取数据失败：{e}")
        return pd.DataFrame()


def select_time(df, start_time='20230101', end_time='20240910'):
    """
    根据指定的时间范围筛选数据。
    参数:
    - df: 包含日期索引的DataFrame
    - start_time: 起始时间 (字符串, 格式 'YYYYMMDD')
    - end_time: 截止时间 (字符串, 格式 'YYYYMMDD')
    返回:
    - 筛选后的DataFrame
    """
    print(f"筛选日期范围: {start_time} 至 {end_time}")
    try:
        start_time = pd.to_datetime(start_time, format='%Y%m%d')
        end_time = pd.to_datetime(end_time, format='%Y%m%d')
    except Exception as e:
        print(f"日期转换错误：{e}")
        return pd.DataFrame()
    df_filtered = df.loc[start_time:end_time]
    print(f"筛选后数据长度: {len(df_filtered)}")
    return df_filtered


##############################################################################
#                       定义全局身份函数替代 lambda
##############################################################################
def identity(x):
    return x

##############################################################################
#                       分类器工厂函数
##############################################################################
def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32)

def get_transformer_classifier(num_features, window_size, class_weights=None):
    if class_weights is not None:
        loss = WeightedCrossEntropyLoss(weight=class_weights)
    else:
        loss = nn.CrossEntropyLoss()
    
    net = NeuralNetClassifier(
        module=TransformerClassifier,
        module__num_features=num_features,
        module__window_size=window_size,
        module__hidden_dim=64,
        module__nhead=8,
        module__num_encoder_layers=2,
        module__dropout=0.1,
        
        max_epochs=10,
        lr=1e-3,
        optimizer=torch.optim.Adam,
        criterion=loss,
        batch_size=64,
        train_split=None,
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return ('transformer', net)

def get_mlp_classifier(input_dim, class_weights=None):
    if class_weights is not None:
        if isinstance(class_weights, torch.Tensor):
            class_weights = class_weights.float()
        loss = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss = nn.CrossEntropyLoss()
    
    net = NeuralNetClassifier(
        module=MLPClassifierModule,
        module__input_dim=input_dim,
        module__hidden_dim=64,
        module__output_dim=2,
        module__dropout=0.5,
        
        criterion=loss,
        optimizer=torch.optim.Adam,
        max_epochs=50,
        lr=1e-3,
        batch_size=64,
        train_split=None,
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return ('mlp', net)

def get_classifier(classifier_name, num_features=None, window_size=10, class_weight=None):
    if classifier_name == '随机森林':
        return ('rf', RandomForestClassifier(
            n_estimators=200,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight=class_weight
        ))
    
    elif classifier_name == '支持向量机':
        return ('svc', SVC(
            probability=True,
            kernel='linear',
            random_state=42,
            class_weight=class_weight
        ))
    
    elif classifier_name == '逻辑回归':
        return ('lr', LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight=class_weight
        ))
    
    elif classifier_name == '梯度提升':
        return ('gb', GradientBoostingClassifier(
            random_state=42
        ))
    
    elif classifier_name == 'Transformer':
        if num_features is None:
            raise ValueError("num_features必须为Transformer模型指定")
        return get_transformer_classifier(num_features, window_size, class_weights=class_weight)
    
    elif classifier_name == '深度学习':
        if num_features is None:
            raise ValueError("num_features必须为MLP模型指定")
        return get_mlp_classifier(num_features, class_weights=class_weight)
    
    else:
        raise ValueError(f"未知的分类器名称: {classifier_name}")

##############################################################################
#                       数据预处理函数
##############################################################################
@st.cache_data
def preprocess_data(data, N, mixture_depth, mark_labels=True, min_features_to_select=10, max_features_for_mixture=50):
    """
    完整保留您原本的 preprocess_data 逻辑，不再省略。
    """
    print("开始预处理数据...")
    data = data.sort_values('TradeDate').copy()
    data.index = pd.to_datetime(data['TradeDate'], format='%Y%m%d')

    # ----------------------------------------
    # 1) 各类技术指标
    # ----------------------------------------
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['Price_MA20_Diff'] = (data['Close'] - data['MA_20']) / data['MA_20']
    data['MA5_MA20_Cross'] = np.where(data['MA_5'] > data['MA_20'], 1, 0)
    data['MA5_MA20_Cross_Diff'] = data['MA5_MA20_Cross'].diff()
    data['Slope_MA5'] = data['MA_5'].diff()

    data['RSI_14'] = compute_RSI(data['Close'], period=14)
    data['MACD'], data['MACD_signal'] = compute_MACD(data['Close'])
    data['MACD_Cross'] = np.where(data['MACD'] > data['MACD_signal'], 1, 0)
    data['MACD_Cross_Diff'] = data['MACD_Cross'].diff()
    data['K'], data['D'] = compute_KD(data['High'], data['Low'], data['Close'], period=14)
    data['Momentum_10'] = compute_momentum(data['Close'], period=10)
    data['ROC_10'] = compute_ROC(data['Close'], period=10)
    data['RSI_Reversal'] = (data['RSI_14'] > 70).astype(int) - (data['RSI_14'] < 30).astype(int)
    data['Reversal_Signal'] = (
        (data['Close'] > data['High'].rolling(window=10).max()).astype(int)
        - (data['Close'] < data['Low'].rolling(window=10).min()).astype(int)
    )

    data['UpperBand'], data['MiddleBand'], data['LowerBand'] = compute_Bollinger_Bands(data['Close'], period=20)
    data['ATR_14'] = compute_ATR(data['High'], data['Low'], data['Close'], period=14)
    data['Volatility_10'] = compute_volatility(data['Close'], period=10)
    data['Bollinger_Width'] = (data['UpperBand'] - data['LowerBand']) / data['MiddleBand']

    if 'Volume' in data.columns:
        data['OBV'] = compute_OBV(data['Close'], data['Volume'])
        data['Volume_Change'] = data['Volume'].pct_change()
        data['VWAP'] = compute_VWAP(data['High'], data['Low'], data['Close'], data['Volume'])
        data['MFI_14'] = compute_MFI(data['High'], data['Low'], data['Close'], data['Volume'], period=14)
        data['CMF_20'] = compute_CMF(data['High'], data['Low'], data['Close'], data['Volume'], period=20)
        data['Chaikin_Osc'] = compute_chaikin_oscillator(data['High'], data['Low'], data['Close'], data['Volume'], short_period=3, long_period=10)
    else:
        data['OBV'] = np.nan
        data['Volume_Change'] = np.nan
        data['VWAP'] = np.nan
        data['MFI_14'] = np.nan
        data['CMF_20'] = np.nan
        data['Chaikin_Osc'] = np.nan

    data['CCI_20'] = compute_CCI(data['High'], data['Low'], data['Close'], period=20)
    data['Williams_%R_14'] = compute_williams_r(data['High'], data['Low'], data['Close'], period=14)
    data['ZScore_20'] = compute_zscore(data['Close'], period=20)
    data['Price_Mean_Diff'] = (data['Close'] - data['Close'].rolling(window=10).mean()) / data['Close'].rolling(window=10).mean()
    data['High_Mean_Diff'] = (data['High'] - data['High'].rolling(window=10).mean()) / data['High'].rolling(window=10).mean()
    data['Low_Mean_Diff'] = (data['Low'] - data['Low'].rolling(window=10).mean()) / data['Low'].rolling(window=10).mean()

    data['Plus_DI'], data['Minus_DI'], data['ADX_14'] = compute_ADX(data['High'], data['Low'], data['Close'], period=14)

    data['TRIX_15'] = compute_TRIX(data['Close'], period=15)
    data['Ultimate_Osc'] = compute_ultimate_oscillator(data['High'], data['Low'], data['Close'], short_period=7, medium_period=14, long_period=28)
    data['PPO'] = compute_PPO(data['Close'], fast_period=12, slow_period=26)
    data['DPO_20'] = compute_DPO(data['Close'], period=20)
    data['KST'], data['KST_signal'] = compute_KST(data['Close'], r1=10, r2=15, r3=20, r4=30, sma1=10, sma2=10, sma3=10, sma4=15)
    data['KAMA_10'] = compute_KAMA(data['Close'], n=10, pow1=2, pow2=30)

    data['Seasonality'] = np.sin(2 * np.pi * data.index.dayofyear / 365)
    data['one'] = 1

    if mark_labels:
        print("寻找局部高点和低点(仅训练阶段)...")
        N = int(N)
        data = identify_low_troughs(data, N)
        data = identify_high_peaks(data, N)
    else:
        # 验证时不标注，但需要占位列
        if 'Peak' in data.columns:
            data.drop(columns=['Peak'], inplace=True)
        if 'Trough' in data.columns:
            data.drop(columns=['Trough'], inplace=True)
        data['Peak'] = 0
        data['Trough'] = 0

    # 添加计数特征
    print("添加计数指标...")
    data['PriceChange'] = data['Close'].diff()
    data['Up'] = np.where(data['PriceChange'] > 0, 1, 0)
    data['Down'] = np.where(data['PriceChange'] < 0, 1, 0)
    data['ConsecutiveUp'] = data['Up'] * (
        data['Up'].groupby((data['Up'] != data['Up'].shift()).cumsum()).cumcount() + 1
    )
    data['ConsecutiveDown'] = data['Down'] * (
        data['Down'].groupby((data['Down'] != data['Down'].shift()).cumsum()).cumcount() + 1
    )

    window_size = 10
    data['Cross_MA5'] = np.where(data['Close'] > data['MA_5'], 1, 0)
    data['Cross_MA5_Count'] = data['Cross_MA5'].rolling(window=window_size).sum()

    if 'Volume' in data.columns:
        data['Volume_MA_5'] = data['Volume'].rolling(window=5).mean()
        data['Volume_Spike'] = np.where(data['Volume'] > data['Volume_MA_5'] * 1.5, 1, 0)
        data['Volume_Spike_Count'] = data['Volume_Spike'].rolling(window=10).sum()
    else:
        data['Volume_Spike_Count'] = np.nan

    # 构建基础因子
    print("构建基础因子...")
    data['Close_MA5_Diff'] = data['Close'] - data['MA_5']
    data['Pch'] = data['Close'] / data['Close'].shift(1) - 1
    data['MA5_MA20_Diff'] = data['MA_5'] - data['MA_20']
    data['RSI_Signal'] = data['RSI_14'] - 50
    data['MACD_Diff'] = data['MACD'] - data['MACD_signal']
    band_range = (data['UpperBand'] - data['LowerBand']).replace(0, np.nan)
    data['Bollinger_Position'] = (data['Close'] - data['MiddleBand']) / band_range
    data['Bollinger_Position'] = data['Bollinger_Position'].fillna(0)
    data['K_D_Diff'] = data['K'] - data['D']

    base_features = [
        'Close_MA5_Diff', 'MA5_MA20_Diff', 'RSI_Signal', 'MACD_Diff',
        'Bollinger_Position', 'K_D_Diff', 'ConsecutiveUp', 'ConsecutiveDown',
        'Cross_MA5_Count', 'Volume_Spike_Count','one','Close','Pch','CCI_20'
    ]

    base_features.extend([
        'Williams_%R_14', 'OBV', 'VWAP','ZScore_20', 'Plus_DI', 'Minus_DI',
        'ADX_14','Bollinger_Width', 'Slope_MA5', 'Volume_Change', 
        'Price_Mean_Diff','High_Mean_Diff','Low_Mean_Diff','MA_5','MA_20','MA_50',
        'MA_200','EMA_5','EMA_20'
    ])

    base_features.extend([
        'MFI_14','CMF_20','TRIX_15','Ultimate_Osc','Chaikin_Osc','PPO',
        'DPO_20','KST','KST_signal','KAMA_10'
    ])

    if 'Volume' in data.columns:
        base_features.append('Volume')

    print("对基础特征进行方差过滤...")
    X_base = data[base_features].fillna(0)
    selector = VarianceThreshold(threshold=0.0001)
    selector.fit(X_base)
    filtered_features = [f for f, s in zip(base_features, selector.get_support()) if s]
    print(f"方差过滤后剩余特征数：{len(filtered_features)}（从{len(base_features)}减少）")
    base_features = filtered_features

    print("对基础特征进行相关性过滤...")
    corr_matrix = data[base_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    base_features = [f for f in base_features if f not in to_drop]
    print(f"相关性过滤后剩余特征数：{len(base_features)}")

    # 生成混合因子
    print(f"生成混合因子，混合深度为 {mixture_depth}...")
    if mixture_depth > 1:
        operators = ['+', '-', '*', '/']
        mixed_features = base_features.copy()
        current_depth_features = base_features.copy()

        for depth in range(2, mixture_depth + 1):
            print(f"生成深度 {depth} 的混合因子...")
            new_features = []
            feature_pairs = combinations(current_depth_features, 2)

            for f1, f2 in feature_pairs:
                for op in operators:
                    new_feature_name = f'({f1}){op}({f2})_d{depth}'
                    try:
                        if op == '+':
                            data[new_feature_name] = data[f1] + data[f2]
                        elif op == '-':
                            data[new_feature_name] = data[f1] - data[f2]
                        elif op == '*':
                            data[new_feature_name] = data[f1] * data[f2]
                        elif op == '/':
                            denom = data[f2].replace(0, np.nan)
                            data[new_feature_name] = data[f1] / denom

                        data[new_feature_name] = data[new_feature_name].replace([np.inf, -np.inf], np.nan).fillna(0)
                        new_features.append(new_feature_name)
                    except Exception as e:
                        print(f"无法计算特征 {new_feature_name}，错误：{e}")

            # 对本轮新特征进行方差和相关性过滤
            if new_features:
                X_new = data[new_features].fillna(0)
                selector = VarianceThreshold(threshold=0.0001)
                selector.fit(X_new)
                new_features = [nf for nf, s in zip(new_features, selector.get_support()) if s]

                if len(new_features) > 1:
                    corr_matrix_new = data[new_features].corr().abs()
                    upper_new = corr_matrix_new.where(np.triu(np.ones(corr_matrix_new.shape), k=1).astype(bool))
                    to_drop_new = [column for column in upper_new.columns if any(upper_new[column] > 0.95)]
                    new_features = [f for f in new_features if f not in to_drop_new]

            mixed_features.extend(new_features)
            current_depth_features = new_features.copy()

        all_features = mixed_features.copy()

        # 使用 PCA 降维
        print("进行 PCA 降维...")
        pca_components = min(100, len(all_features))
        pca = PCA(n_components=pca_components)
        X_mixed = data[all_features].fillna(0).values
        X_mixed_pca = pca.fit_transform(X_mixed)
        pca_feature_names = [f'PCA_{i}' for i in range(pca_components)]
        for i in range(pca_components):
            data[pca_feature_names[i]] = X_mixed_pca[:, i]
        all_features = pca_feature_names
    else:
        all_features = base_features.copy()

    print(f"最终特征数量：{len(all_features)}")

    # 删除缺失值
    required_cols = [
        'Close_MA5_Diff', 'MA5_MA20_Diff', 'RSI_Signal', 'MACD_Diff',
        'Bollinger_Position', 'K_D_Diff'
    ]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"列 {col} 未被创建，请检查数据和计算步骤。")

    print("删除缺失值...")
    initial_length = len(data)
    data = data.dropna().copy()
    final_length = len(data)
    print(f"数据预处理前长度: {initial_length}, 数据预处理后长度: {final_length}")

    return data, all_features

##############################################################################
#                       优化阈值函数
##############################################################################
@st.cache_data
def optimize_threshold(y_true, y_proba):
    """
    优化分类阈值以最大化 F1 分数。

    参数:
    - y_true: 真实标签
    - y_proba: 预测概率

    返回:
    - 最佳阈值
    """
    best_thresh = 0.5
    best_f1 = -1

    for thresh in np.linspace(0, 1, 101):
        y_pred_temp = (y_proba > thresh).astype(int)
        score = f1_score(y_true, y_pred_temp)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return best_thresh

##############################################################################
#                       创建序列数据函数
##############################################################################
@st.cache_data
def create_sequences(X, y=None, window_size=10):
    """
    创建时间序列数据的函数。

    参数:
    - X: 特征数组
    - y: 标签数组（可选）
    - window_size: 窗口大小

    返回:
    - sequences: 序列数据
    - labels: 标签（如果提供了 y）
    """
    sequences = []
    if y is not None:
        labels = []
    
    for i in range(window_size, len(X) + 1):
        seq_x = X[i - window_size:i].astype(np.float32)
        sequences.append(seq_x)
        
        if y is not None:
            seq_y = y[i - 1]
            labels.append(seq_y)
    
    sequences = np.array(sequences)
    
    if y is not None:
        labels = np.array(labels, dtype=np.int64)
        return sequences, labels
    return sequences

##############################################################################
#                       训练模型函数
##############################################################################
def train_model_for_label(
    df, N, label_column, all_features, classifier_name, 
    n_features_selected, window_size=10, oversample_method='SMOTE', class_weight=None
):
    print(f"开始训练 {label_column} 模型...")
    data = df.copy()
    
    # 特征相关性过滤（可选，根据需要）
    print("开始特征相关性过滤...")
    corr_matrix = data[all_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if to_drop:
        print(f"检测到高相关特征 {len(to_drop)} 个，将进行剔除。")
    else:
        print("未检测到高相关特征。")
    all_features_filtered = [f for f in all_features if f not in to_drop]
    X = data[all_features_filtered]
    print(f"过滤后特征数量：{len(all_features_filtered)}")

    # 数据标准化
    print("标准化数据...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # 选择并应用过采样方法
    print(f"应用过采样方法：{oversample_method}")
    if oversample_method == '过采样':
        sampler = SMOTE(random_state=42)
    elif oversample_method == '类别权重':
        sampler = None
    else:
        raise ValueError(f"未知的过采样方法: {oversample_method}")

    if sampler is not None:
        if classifier_name == 'Transformer':
            print("为 Transformer 创建序列数据...")
            X_seq, y_seq = create_sequences(X_scaled, data[label_column].astype(np.int64), window_size=window_size)
            print(f"序列数据形状: X={X_seq.shape}, y={y_seq.shape}")
            
            # 重塑数据以适应 SMOTE
            X_reshaped = X_seq.reshape(X_seq.shape[0], -1)
            print("对序列数据进行过采样处理...")
            X_resampled, y_resampled = sampler.fit_resample(X_reshaped, y_seq)
            
            X_resampled = X_resampled.reshape(-1, window_size, X_seq.shape[2])
            print(f"过采样后数据形状: X={X_resampled.shape}, y={y_resampled.shape}")
        else:
            print("对数据进行过采样处理...")
            X_resampled, y_resampled = sampler.fit_resample(X_scaled, data[label_column].astype(np.int64))
            print(f"数据形状: X={X_resampled.shape}, y={y_resampled.shape}")
    else:
        print("不进行过采样，使用原始数据。")
        X_resampled, y_resampled = X_scaled, data[label_column].astype(np.int64).values

    # 计算类别权重
    if oversample_method == '类别权重':
        class_weights_array = get_class_weights(y_resampled)
        if isinstance(class_weights_array, torch.Tensor):
            class_weights_array = class_weights_array.float()
    else:
        class_weights_array = class_weight

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, 
        random_state=42, stratify=y_resampled
    )
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    # 获取分类器
    num_features = X_train.shape[-1] if classifier_name == 'Transformer' else X_train.shape[1]
    if classifier_name == 'Transformer':
        clf_name, clf = get_transformer_classifier(
            num_features=num_features,
            window_size=window_size,
            class_weights=class_weights_array
        )
    elif classifier_name == '深度学习':
        clf_name, clf = get_mlp_classifier(
            input_dim=num_features,
            class_weights=class_weights_array
        )
    else:
        clf_name, clf = get_classifier(
            classifier_name,
            num_features=num_features,
            window_size=window_size,
            class_weight='balanced' if oversample_method == '类别权重' else class_weight
        )

    # 网格搜索
    print(f"正在为分类器 {clf_name} 进行 GridSearchCV 调参...")
    param_grid = {}
    if clf_name == 'rf':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif clf_name == 'svc':
        param_grid = {
            'C': [0.1, 1],
            'gamma': ['scale', 'auto']
        }
    elif clf_name == 'lr':
        param_grid = {
            'C': [0.01, 0.1, 1],
            'penalty': ['l2']
        }
    elif clf_name == 'gb':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    elif clf_name in ['transformer', 'mlp']:
        param_grid = {
            'lr': [1e-3],
            'max_epochs': [10]
        }
        if clf_name == 'transformer':
            param_grid['module__hidden_dim'] = [64]
    else:
        param_grid = {}

    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=3,
        n_jobs=1 if clf_name in ['transformer', 'mlp'] else -1,
        scoring='f1',
        verbose=1,
        error_score='raise'
    )

    try:
        grid_search.fit(X_train, y_train)
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳得分: {grid_search.best_score_:.4f}")
        best_estimator = grid_search.best_estimator_
    except Exception as e:
        print(f"GridSearchCV 失败: {e}")
        raise

    # 特征选择
    print("开始特征选择...")
    feature_selector = None
    selected_features = all_features_filtered.copy()

    if clf_name in ['rf', 'gb'] and n_features_selected != 'auto':
        feature_selector = RFE(
            estimator=best_estimator,
            n_features_to_select=int(n_features_selected),
            step=1
        )
        # transformer 不同结构无需 RFE
        if not clf_name == 'transformer':
            feature_selector.fit(X_train, y_train)
            selected_features = [
                all_features_filtered[i] 
                for i in range(len(all_features_filtered)) 
                if feature_selector.support_[i]
            ]
            print(f"RFE选择的特征数量：{len(selected_features)}")
    else:
        print(f"{clf_name} 不进行特征选择，使用全部特征")
        feature_selector = FunctionTransformer(func=identity, validate=False)

    # 评估模型
    if isinstance(best_estimator, NeuralNetClassifier):
        if feature_selector is not None and not isinstance(feature_selector, FunctionTransformer):
            X_test_selected = feature_selector.transform(X_test)
            logits = best_estimator.predict_proba(X_test_selected)
        else:
            logits = best_estimator.predict_proba(X_test)
            
        if isinstance(logits, np.ndarray) and logits.ndim == 2:
            y_proba = logits[:, 1]
        else:
            y_proba = F.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    else:
        if feature_selector is not None:
            X_test_selected = feature_selector.transform(X_test)
            y_proba = best_estimator.predict_proba(X_test_selected)[:, 1]
        else:
            y_proba = best_estimator.predict_proba(X_test)[:, 1]

    best_thresh = optimize_threshold(y_test, y_proba)
    y_pred = (y_proba > best_thresh).astype(int)

    # 计算评估指标
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n模型评估结果：")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"MCC: {mcc:.4f}")

    metrics = {
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc,
        'Precision': precision,
        'Recall': recall,
        'MCC': mcc
    }

    return (
        best_estimator, 
        scaler, 
        feature_selector, 
        selected_features, 
        all_features_filtered, 
        grid_search.best_score_, 
        metrics, 
        best_thresh
    )

##############################################################################
#                       并行训练 Peak & Trough 主函数
##############################################################################
@st.cache_resource
def train_model(
    df_preprocessed, N, all_features, classifier_name, 
    mixture_depth, n_features_selected, oversample_method, window_size=10
):
    print("开始训练模型...")
    data = df_preprocessed.copy()
    print(f"预处理后数据长度: {len(data)}")

    labels = ['Peak', 'Trough']
    
    results = Parallel(n_jobs=-1)(
        delayed(train_model_for_label)(
            data, N, label, all_features, classifier_name, 
            n_features_selected, window_size, 
            oversample_method, 
            class_weight='balanced' if oversample_method == '类别权重' else None
        )
        for label in labels
    )

    peak_results = results[0]
    trough_results = results[1]
    return peak_results, trough_results

##############################################################################
#                       平滑验证函数
##############################################################################
def smooth_predictions(pred_series, min_days_between_predictions=20):
    peaks = []
    last_peak = -min_days_between_predictions
    for i in range(len(pred_series)):
        if pred_series[i] == 1 and (i - last_peak) >= min_days_between_predictions:
            peaks.append(i)
            last_peak = i
    pred = np.zeros(len(pred_series))
    pred[peaks] = 1
    return pred

##############################################################################
#                       验证新数据的函数
##############################################################################
@st.cache_data
def predict_new_data(new_df,
                     _peak_model,
                     _peak_scaler,
                     _peak_selector,
                     all_features_peak,
                     peak_threshold,
                     _trough_model,
                     _trough_scaler,
                     _trough_selector,
                     all_features_trough,
                     trough_threshold,
                     N, mixture_depth=3, min_days_between_predictions=20):
    print("开始验证新数据...")
    
    # 预处理数据（mark_labels=False）
    data_preprocessed, _ = preprocess_data(new_df, N, mixture_depth=mixture_depth, mark_labels=False)
    print(f"预处理后数据长度: {len(data_preprocessed)}")

    # ========== Peak 验证 ==========
    print("\n开始Peak验证...")
    missing_features_peak = [f for f in all_features_peak if f not in data_preprocessed.columns]
    if missing_features_peak:
        print(f"填充缺失特征(Peak): {missing_features_peak}")
        for feature in missing_features_peak:
            data_preprocessed[feature] = 0
            
    X_new_peak = data_preprocessed[all_features_peak].fillna(0)
    X_new_peak_scaled = _peak_scaler.transform(X_new_peak).astype(np.float32)
    print(f"Peak数据形状: {X_new_peak_scaled.shape}")

    if isinstance(_peak_model, NeuralNetClassifier) and \
       isinstance(_peak_model.module_, TransformerClassifier):
        print("创建Peak序列数据...")
        X_new_seq_peak = create_sequences(X_new_peak_scaled, y=None, window_size=10)  # 固定为10
        print(f"Peak序列数据形状: {X_new_seq_peak.shape}")

        batch_size = 64
        predictions = []
        _peak_model.module_.eval()
        
        with torch.no_grad():
            for i in range(0, len(X_new_seq_peak), batch_size):
                batch = torch.from_numpy(X_new_seq_peak[i:i+batch_size]).float()
                batch = batch.to(_peak_model.device)
                outputs = _peak_model.module_(batch)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                predictions.append(probs.cpu().numpy())

        all_probas = np.concatenate(predictions)
        peak_probas = np.zeros(len(data_preprocessed))
        peak_probas[10-1:] = all_probas  # window_size=10
        
    else:
        if isinstance(_peak_model, NeuralNetClassifier):
            if _peak_selector is not None and not isinstance(_peak_selector, FunctionTransformer):
                X_new_peak_selected = _peak_selector.transform(X_new_peak_scaled)
                logits = _peak_model.predict_proba(X_new_peak_selected)
            else:
                logits = _peak_model.predict_proba(X_new_peak_scaled)
            if logits.ndim == 2:
                peak_probas = logits[:, 1]
            else:
                peak_probas = torch.sigmoid(torch.tensor(logits)).numpy()
        else:
            if _peak_selector is not None:
                X_new_peak_selected = _peak_selector.transform(X_new_peak_scaled)
                peak_probas = _peak_model.predict_proba(X_new_peak_selected)[:, 1]
            else:
                peak_probas = _peak_model.predict_proba(X_new_peak_scaled)[:, 1]

    peak_preds = (peak_probas > peak_threshold).astype(int)
    data_preprocessed['Peak_Probability'] = peak_probas
    data_preprocessed['Peak_Prediction'] = peak_preds

    # ========== Trough 验证 ==========
    print("\n开始Trough验证...")
    missing_features_trough = [f for f in all_features_trough if f not in data_preprocessed.columns]
    if missing_features_trough:
        print(f"填充缺失特征(Trough): {missing_features_trough}")
        for feature in missing_features_trough:
            data_preprocessed[feature] = 0
            
    X_new_trough = data_preprocessed[all_features_trough].fillna(0)
    X_new_trough_scaled = _trough_scaler.transform(X_new_trough).astype(np.float32)
    print(f"Trough数据形状: {X_new_trough_scaled.shape}")

    if isinstance(_trough_model, NeuralNetClassifier) and \
       isinstance(_trough_model.module_, TransformerClassifier):
        print("创建Trough序列数据...")
        X_new_seq_trough = create_sequences(X_new_trough_scaled, y=None, window_size=10)  # 固定为10
        print(f"Trough序列数据形状: {X_new_seq_trough.shape}")

        batch_size = 64
        predictions = []
        _trough_model.module_.eval()
        
        with torch.no_grad():
            for i in range(0, len(X_new_seq_trough), batch_size):
                batch = torch.from_numpy(X_new_seq_trough[i:i+batch_size]).float()
                batch = batch.to(_trough_model.device)
                outputs = _trough_model.module_(batch)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                predictions.append(probs.cpu().numpy())

        all_probas = np.concatenate(predictions)
        trough_probas = np.zeros(len(data_preprocessed))
        trough_probas[10-1:] = all_probas  # window_size=10
        
    else:
        if isinstance(_trough_model, NeuralNetClassifier):
            if _trough_selector is not None and not isinstance(_trough_selector, FunctionTransformer):
                X_new_trough_selected = _trough_selector.transform(X_new_trough_scaled)
                logits = _trough_model.predict_proba(X_new_trough_selected)
            else:
                logits = _trough_model.predict_proba(X_new_trough_scaled)
            if logits.ndim == 2:
                trough_probas = logits[:, 1]
            else:
                trough_probas = torch.sigmoid(torch.tensor(logits)).numpy()
        else:
            if _trough_selector is not None:
                X_new_trough_selected = _trough_selector.transform(X_new_trough_scaled)
                trough_probas = _trough_model.predict_proba(X_new_trough_selected)[:, 1]
            else:
                trough_probas = _trough_model.predict_proba(X_new_trough_scaled)[:, 1]

    trough_preds = (trough_probas > trough_threshold).astype(int)
    data_preprocessed['Trough_Probability'] = trough_probas
    data_preprocessed['Trough_Prediction'] = trough_preds

    # 后处理：指定天数内不重复验证
    print("\n进行后处理...")
    data_preprocessed.index = data_preprocessed.index.astype(str)
    
    # 使用平滑预测函数
    data_preprocessed['Peak_Prediction'] = smooth_predictions(data_preprocessed['Peak_Prediction'], min_days_between_predictions)
    data_preprocessed['Trough_Prediction'] = smooth_predictions(data_preprocessed['Trough_Prediction'], min_days_between_predictions)

    return data_preprocessed

##############################################################################
#                       绘图函数（使用Plotly）
##############################################################################
def plot_candlestick_plotly(data, symbol_code, start_date, end_date, peaks=None, troughs=None, prediction=False, selected_classifiers=None):
    if prediction and selected_classifiers:
        classifiers_str = ", ".join(selected_classifiers)
        title = f"{symbol_code} {start_date} 至 {end_date} 基础模型: {classifiers_str}"
    else:
        title = f"{symbol_code} {start_date} 至 {end_date}"

    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(f"data.index 无法转换为日期格式: {e}")
    data.index = data.index.strftime('%Y-%m-%d')

    if peaks is not None and not peaks.empty:
        if not isinstance(peaks.index, pd.DatetimeIndex):
            try:
                peaks.index = pd.to_datetime(peaks.index)
            except Exception as e:
                raise ValueError(f"peaks.index 无法转换为日期格式: {e}")
        peaks.index = peaks.index.strftime('%Y-%m-%d')

    if troughs is not None and not troughs.empty:
        if not isinstance(troughs.index, pd.DatetimeIndex):
            try:
                troughs.index = pd.to_datetime(troughs.index)
            except Exception as e:
                raise ValueError(f"troughs.index 无法转换为日期格式: {e}")
        troughs.index = troughs.index.strftime('%Y-%m-%d')

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "candlestick"}],[{"type": "bar"}]]
    )

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol_code,
        increasing=dict(line=dict(color='red')),
        decreasing=dict(line=dict(color='green')),
        hoverinfo='x+y+text',
    ), row=1, col=1)

    if 'Volume' in data.columns:
        volume_colors = ['red' if row['Close'] > row['Open'] else 'green' for _, row in data.iterrows()]
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color=volume_colors,
            name='成交量',
            hoverinfo='x+y'
        ), row=2, col=1)

    if peaks is not None and not peaks.empty:
        marker_y_peaks = peaks['High'] * 1.02
        marker_x_peaks = peaks.index
        color_peak = 'green'
        label_peak = '局部高点' if not prediction else '验证高点'
        fig.add_trace(go.Scatter(
            x=marker_x_peaks,
            y=marker_y_peaks,
            mode='text',
            text='W',
            textfont=dict(color=color_peak, size=20),
            name=label_peak
        ), row=1, col=1)

    if troughs is not None and not troughs.empty:
        marker_y_troughs = troughs['Low'] * 0.98
        marker_x_troughs = troughs.index
        color_trough = 'red'
        label_trough = '局部低点' if not prediction else '验证低点'
        fig.add_trace(go.Scatter(
            x=marker_x_troughs,
            y=marker_y_troughs,
            mode='text',
            text='D',
            textfont=dict(color=color_trough, size=20),
            name=label_trough
        ), row=1, col=1)

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="日期",
            type="category",
            tickangle=45,
            tickmode="auto",
            nticks=10
        ),
        xaxis2=dict(
            title="日期",
            type="category",
            tickangle=45,
            tickmode="auto",
            nticks=10
        ),
        yaxis_title="价格",
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        height=800,
        font=dict(
            family="Microsoft YaHei, SimHei",
            size=14,
            color="black"
        )
    )
    return fig

##############################################################################
#                       下载按钮函数
##############################################################################
def download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # 编码为base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

##############################################################################
#                       Streamlit 主函数（布局调整后）
##############################################################################
def main():
    st.set_page_config(page_title="指数局部高低点验证", layout="wide")
    st.title("指数局部高低点验证")

    # 使用 Sidebar 分区，将模型训练与模型验证分别放在不同的导航选项中
    menu = st.sidebar.radio("导航", ["📈 模型训练", "🔍 模型验证"])

    if menu == "📈 模型训练":
        train_section()
    elif menu == "🔍 模型验证":
        predict_section()

##############################################################################
#                       训练部分函数
##############################################################################
def train_section():
    st.header("模型训练流程")
    st.markdown("""
    **机器学习主要步骤：**  
    1. **数据标注**：标注高低点，作为训练目标。  
    2. **选择基础模型**：推荐使用深度学习和梯度提升。  
    3. **均衡类别**：训练时高低点与非高低点数量需接近1：1，需过采样或提高高低点权重。  
    4. **特征选择**：特征即指标，内置32个，混合后将生成更多指标。按与行情相关程度由高到底排序，可手动选择数量。
    """)

    with st.expander("🛠️ 数据与参数设置", expanded=True):
        with st.form("train_form"):
            # ========== 修改部分开始 ==========
            symbol_type_display = st.radio(
                "1️⃣ 代码类型",
                options=["股票", "指数"],
                index=1,  # 默认选择“指数”
                help="选择输入的是股票代码还是指数代码。"
            )

            # 将中文选择转换为英文
            symbol_type = 'stock' if symbol_type_display == "股票" else 'index'

            # 动态设置默认 symbol_code
            if symbol_type == "stock":
                default_symbol_code = "000001.SZ"
            else:
                default_symbol_code = "000001.SH"

            symbol_code = st.text_input(
                "2️⃣ 股票或指数代码", 
                value=default_symbol_code, 
                help="请输入股票代码（如 000001.SZ）或指数代码（如 000300.SH）。"
            )
            # ========== 修改部分结束 ==========

            # 训练集日期
            st.markdown("📅 **训练集日期**")
            col_train_date1, col_train_date2 = st.columns(2)
            with col_train_date1:
                start_date = st.date_input(
                    "训练开始日期", 
                    datetime.strptime("2000-01-01", "%Y-%m-%d"), 
                    key='train_start_date',
                    help="选择训练数据的开始日期。"
                )
            with col_train_date2:
                end_date = st.date_input(
                    "训练结束日期", 
                    datetime.strptime("2020-12-31", "%Y-%m-%d"), 
                    key='train_end_date',
                    help="选择训练数据的结束日期。"
                )

            N = st.number_input(
                "3️⃣ 标注高低点间的最小间隔 (N)", 
                min_value=1, 
                max_value=1000000, 
                value=30, 
                help="用于数据预处理的窗口长度，决定如何标注高低点。"
            )

            # 选择基础模型
            available_classifiers = ['随机森林', '支持向量机', '逻辑回归', '梯度提升', 'Transformer', '深度学习']
            classifier_name = st.selectbox(
                "4️⃣ 选择基础模型", 
                available_classifiers, 
                help="选择用于训练的分类器模型。"
            )

            # 因子混合深度
            mixture_depth = st.slider(
                "5️⃣ 因子混合深度", 
                min_value=1, 
                max_value=3, 
                value=1, 
                help="选择因子混合的深度。"
            )

            # 过采样方法选择
            oversample_methods = [
                '过采样',
                '类别权重'
            ]
            oversample_method = st.selectbox(
                "6️⃣ 处理类别不均衡的方法", 
                oversample_methods, 
                help="选择用于处理类别不均衡的方法。"
            )

            # 特征选择
            st.markdown("🔍 **特征选择**")
            auto_feature = st.checkbox(
                "自动选择特征数量（仅对随机森林、梯度提升有效）", 
                value=True
            )
            if auto_feature:
                n_features_selected = 'auto'
            else:
                n_features_selected = st.number_input(
                    "选择特征数量", 
                    min_value=1, 
                    max_value=1000, 
                    value=20, 
                    help="手动选择特征的数量。"
                )

            submit_train = st.form_submit_button("提交参数")
        
        if submit_train:
            st.session_state['train_params'] = {
                'symbol_type': symbol_type,  # 使用英文的 'stock' 或 'index'
                'symbol_code': symbol_code,
                'start_date': start_date,
                'end_date': end_date,
                'N': N,
                'classifier_name': classifier_name,
                'mixture_depth': mixture_depth,
                'oversample_method': oversample_method,
                'n_features_selected': n_features_selected
            }
            st.success("参数已提交，请点击下方『训练模型』按钮开始训练。")

            # ========== 获取并显示名称 ==========
            try:
                if symbol_type == "stock":
                    stock_info = pro.stock_basic(ts_code=symbol_code, fields='ts_code,name')
                    if not stock_info.empty:
                        stock_name = stock_info.iloc[0]['name']
                        st.markdown(f"**股票名称：** {stock_name}")
                    else:
                        st.warning("无法获取股票名称，请检查股票代码。")
                else:
                    index_info = pro.index_basic(ts_code=symbol_code, fields='ts_code,name')
                    if not index_info.empty:
                        index_name = index_info.iloc[0]['name']
                        st.markdown(f"**指数名称：** {index_name}")
                    else:
                        st.warning("无法获取指数名称，请检查指数代码。")
            except Exception as e:
                st.error(f"获取名称失败：{e}")

            # ========== 调试输出：尝试获取数据并显示 ==========
            try:
                #st.write("正在尝试获取数据...")
                data = read_day_from_tushare(symbol_code, symbol_type=symbol_type)
                st.write(f"获取到的数据行数: {len(data)}")
                st.write(f"数据预处理完成，请点击按钮开始训练")
                if data.empty:
                    st.warning("获取的数据为空，请检查代码类型和代码是否正确。")
            except Exception as e:
                st.error(f"获取数据时发生错误：{e}")

    st.markdown("---")
    st.subheader("🚀 开始训练")
    st.markdown("请点击下方按钮开始训练模型。")
    
    if st.button("训练模型"):
        if 'train_params' not in st.session_state or not st.session_state['train_params']:
            st.error("请在上方提交训练参数。")
        else:
            params = st.session_state['train_params']
            symbol_type = params['symbol_type']
            symbol_code = params['symbol_code']
            start_date = params['start_date']
            end_date = params['end_date']
            N = params['N']
            classifier_name = params['classifier_name']
            mixture_depth = params['mixture_depth']
            oversample_method = params['oversample_method']
            n_features_selected = params['n_features_selected']

            # 参数验证
            if not all([symbol_code, N, mixture_depth, classifier_name]):
                st.error("请填写所有训练参数。")
            elif start_date > end_date:
                st.error("开始日期不能晚于结束日期。")
            else:
                with st.spinner("正在读取数据并进行预处理..."):
                    try:
                        # 读取数据
                        data = read_day_from_tushare(symbol_code, symbol_type=symbol_type)
                        if data.empty:
                            st.error("通过 Tushare 获取的数据为空，请检查代码类型和代码。")
                            st.stop()
                        
                        # 根据日期范围截取数据
                        df = select_time(data, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
                        if df.empty:
                            st.error("训练集为空，请检查日期范围和代码。")
                            st.stop()
                        
                        # 预处理数据
                        df_preprocessed, all_features = preprocess_data(
                            df, N, mixture_depth, mark_labels=True
                        )
                        
                        st.success("数据预处理完成")
                        
                        # ========== 显示筛选后的数据行数和特征数量 ==========
                        num_rows = len(df_preprocessed)
                        num_features = len(all_features)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**筛选后的数据行数：** {num_rows}")
                        with col2:
                            st.write(f"**特征数量：** {num_features}")
                        
                    except AssertionError as ae:
                        st.error(f"数据处理断言失败：{ae}")
                        st.stop()
                    except Exception as e:
                        st.error(f"数据处理失败：{e}")
                        st.stop()

                # ========== 显示标注好的图表 ==========
                st.subheader("📊 预处理后的标注图表")
                try:
                    peaks = df_preprocessed[df_preprocessed['Peak'] == 1] if 'Peak' in df_preprocessed.columns else pd.DataFrame()
                    troughs = df_preprocessed[df_preprocessed['Trough'] == 1] if 'Trough' in df_preprocessed.columns else pd.DataFrame()
                    symbol_display = symbol_code
                    fig_initial = plot_candlestick_plotly(
                        df_preprocessed, symbol_display, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"),
                        peaks=peaks, troughs=troughs, prediction=False, selected_classifiers=None
                    )
                    st.plotly_chart(fig_initial, use_container_width=True)
                except Exception as e:
                    st.error(f"绘制标注图表失败：{e}")
                    st.stop()

                # ========== 开始模型训练 ==========
                with st.spinner("模型训练中，请稍候..."):
                    try:
                        # 并行训练 peak / trough
                        peak_results, trough_results = train_model(
                            df_preprocessed, N, all_features, classifier_name, 
                            mixture_depth, n_features_selected, oversample_method
                        )
                        
                        (peak_model, peak_scaler, peak_selector, peak_selected_features, all_features_peak, peak_best_score,
                         peak_metrics, peak_threshold) = peak_results
                        (trough_model, trough_scaler, trough_selector, trough_selected_features, all_features_trough,
                         trough_best_score, trough_metrics, trough_threshold) = trough_results
                        
                        st.success("模型训练完成！")
                        
                        # 训练结果展示
                        st.subheader("📈 训练结果与评估指标")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("#### 高点识别")
                            st.write(f"**最佳得分：** {peak_best_score:.2f}")
                            st.write("**评估指标：**")
                            # 将指标保留2位小数并翻译
                            metrics_peak = {k: f"{v:.2f}" for k, v in peak_metrics.items()}
                            metrics_peak_cn = {
                                'ROC AUC': 'ROC AUC',
                                'PR AUC': 'PR AUC',
                                'Precision': '精确率',
                                'Recall': '召回率',
                                'MCC': 'MCC'
                            }
                            metrics_peak_cn_translated = {metrics_peak_cn[k]: v for k, v in metrics_peak.items()}
                            st.table(pd.DataFrame(metrics_peak_cn_translated, index=[0]))
                        with c2:
                            st.markdown("#### 低点识别")
                            st.write(f"**最佳得分：** {trough_best_score:.2f}")
                            st.write("**评估指标：**")
                            # 将指标保留2位小数并翻译
                            metrics_trough = {k: f"{v:.2f}" for k, v in trough_metrics.items()}
                            metrics_trough_cn = {
                                'ROC AUC': 'ROC AUC',
                                'PR AUC': 'PR AUC',
                                'Precision': '精确率',
                                'Recall': '召回率',
                                'MCC': 'MCC'
                            }
                            metrics_trough_cn_translated = {metrics_trough_cn[k]: v for k, v in metrics_trough.items()}
                            st.table(pd.DataFrame(metrics_trough_cn_translated, index=[0]))
                        
                    except Exception as e:
                        st.error(f"模型训练失败：{e}")
                        st.stop()
                
                # ========== 缓存模型到 session_state ==========
                try:
                    st.session_state['peak_model'] = peak_model
                    st.session_state['peak_scaler'] = peak_scaler
                    st.session_state['peak_selector'] = peak_selector
                    st.session_state['all_features_peak'] = all_features_peak
                    st.session_state['peak_threshold'] = peak_threshold
                    
                    st.session_state['trough_model'] = trough_model
                    st.session_state['trough_scaler'] = trough_scaler
                    st.session_state['trough_selector'] = trough_selector
                    st.session_state['all_features_trough'] = all_features_trough
                    st.session_state['trough_threshold'] = trough_threshold
                except Exception as e:
                    st.error(f"缓存模型失败：{e}")
                    st.stop()



##############################################################################
#                       验证部分函数
##############################################################################
def predict_section():
    st.header("模型验证流程")
    st.markdown("""
    **在此部分可完成：**  
    1. **读取新数据并进行相同的预处理**  
    2. **调用训练好的模型进行高低点验证**   
    3. **可视化验证结果** 
    """)

    with st.expander("🔧 验证数据设置", expanded=True):
        with st.form("predict_form"):
            # 验证区间
            st.markdown("📅 **验证区间**")
            col_pred_date1, col_pred_date2 = st.columns(2)
            with col_pred_date1:
                start_new_date = st.date_input(
                    "验证开始日期", 
                    datetime.strptime("2021-01-01", "%Y-%m-%d"), 
                    key='pred_start_date',
                    help="选择验证数据的开始日期。"
                )
            with col_pred_date2:
                end_new_date = st.date_input(
                    "验证结束日期", 
                    datetime.today(), 
                    key='pred_end_date',
                    help="选择验证数据的结束日期。"
                )

            # 其他验证参数
            st.markdown("⚙️ **其他验证参数**")
            # 添加最小验证间隔天数
            min_days_between_predictions = st.number_input(
                "7️⃣ 最小验证间隔天数",  # 修改编号
                min_value=1,
                max_value=365,
                value=20,
                help="在指定的天数内不允许重复验证（即相同类别的预测点之间至少相隔多少天）。"
            )

            submit_predict = st.form_submit_button("提交验证参数")
        
        if submit_predict:
            st.session_state['predict_params'] = {
                'start_new_date': start_new_date,
                'end_new_date': end_new_date,
                'min_days_between_predictions': min_days_between_predictions
            }
            st.success("验证参数已提交，请点击下方『调用模型进行验证』按钮开始验证。")

    st.markdown("---")
    st.subheader("🔄 开始验证")
    st.markdown("请点击下方按钮进行验证，并可查看验证后高低点可视化结果。")

    if st.button("调用模型进行验证"):
        if not all([
            'peak_model' in st.session_state, 
            'trough_model' in st.session_state,
            'predict_params' in st.session_state,
            'train_params' in st.session_state  # 确保训练参数存在以获取代码类型
        ]):
            st.error("请先在左侧『模型训练』部分完成模型训练后再进行验证。")
        else:
            params = st.session_state['predict_params']
            start_new_date = params['start_new_date']
            end_new_date = params['end_new_date']
            min_days_between_predictions = params['min_days_between_predictions']

            # 参数验证
            if start_new_date > end_new_date:
                st.error("验证开始日期不能晚于验证结束日期。")
            else:
                with st.spinner("正在读取验证数据并进行预处理..."):
                    try:
                        # 获取训练参数以获取 symbol_type 和 symbol_code
                        train_params = st.session_state['train_params']
                        symbol_type = train_params['symbol_type']
                        symbol_code = train_params['symbol_code']

                        # 显示当前使用的 symbol_type 和 symbol_code 以确认
                        st.markdown(f"**使用的代码类型：** {'股票' if symbol_type == 'stock' else '指数'}")
                        st.markdown(f"**使用的代码：** {symbol_code}")

                        # 读取数据
                        data = read_day_from_tushare(symbol_code, symbol_type=symbol_type)
                        if data.empty:
                            st.error("通过 Tushare 获取的数据为空，请检查代码类型和代码。")
                            st.stop()
                        
                        # 截取验证区间
                        new_df = select_time(data, start_new_date.strftime("%Y%m%d"), end_new_date.strftime("%Y%m%d"))
                        if new_df.empty:
                            st.error("验证集为空，请检查日期范围和代码。")
                            st.stop()
                        
                        # 调用验证
                        result = predict_new_data(
                            new_df,
                            _peak_model=st.session_state['peak_model'], 
                            _peak_scaler=st.session_state['peak_scaler'], 
                            _peak_selector=st.session_state['peak_selector'], 
                            all_features_peak=st.session_state['all_features_peak'], 
                            peak_threshold=st.session_state['peak_threshold'],
                            _trough_model=st.session_state['trough_model'], 
                            _trough_scaler=st.session_state['trough_scaler'], 
                            _trough_selector=st.session_state['trough_selector'], 
                            all_features_trough=st.session_state['all_features_trough'], 
                            trough_threshold=st.session_state['trough_threshold'],
                            N=st.session_state['train_params']['N'], 
                            mixture_depth=st.session_state['train_params']['mixture_depth'], 
                            min_days_between_predictions=min_days_between_predictions
                        )
                        
                        st.success("验证完成！")
                        
                        # ========== 显示验证结果的图表 ==========
                        st.subheader("📊 验证结果可视化")
                        peaks_pred = result[result['Peak_Prediction'] == 1]
                        troughs_pred = result[result['Trough_Prediction'] == 1]
                        symbol_display = symbol_code
                        fig_pred = plot_candlestick_plotly(
                            result, symbol_display, start_new_date.strftime("%Y%m%d"), end_new_date.strftime("%Y%m%d"),
                            peaks=peaks_pred, troughs=troughs_pred, prediction=True, 
                            selected_classifiers=[st.session_state['train_params']['classifier_name']]
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)

                        # ========== 显示验证结果的数据表 ==========
                        st.subheader("📄 验证结果（仅显示验证到的高低点）")
                        filtered_result = result[
                            (result['Peak_Prediction'] == 1) | (result['Trough_Prediction'] == 1)
                        ]
                        
                        if filtered_result.empty:
                            st.info("没有验证到高点或低点。")
                        else:
                            result_table = filtered_result[['TradeDate', 'Peak_Prediction', 'Peak_Probability',
                                                            'Trough_Prediction', 'Trough_Probability']].copy()
                            result_table = result_table.rename(columns={
                                'Peak_Prediction': '高点',
                                'Peak_Probability': '高点概率',
                                'Trough_Prediction': '低点',
                                'Trough_Probability': '低点概率'
                            })
                            # 保留2位小数
                            result_table['高点概率'] = result_table['高点概率'].astype(float).round(2)
                            result_table['低点概率'] = result_table['低点概率'].astype(float).round(2)
                            
                            AgGrid(result_table, fit_columns_on_grid_load=True, height=300, theme='streamlit')
                            
                            # 添加下载按钮
                            st.markdown(download_link(result_table, 'validation_results.csv', '📥 下载验证结果'), unsafe_allow_html=True)
                        
                    except AssertionError as ae:
                        st.error(f"验证数据处理断言失败：{ae}")
                        st.stop()
                    except Exception as e:
                        st.error(f"验证失败：{e}")
                        st.stop()

##############################################################################
#                       其他必要函数和导入
##############################################################################
# 请确保您已经在 function.py 中定义了所有必要的计算函数，如 compute_RSI、compute_MACD 等。
# 同时，请确保 models.py 中定义了所有自定义模型类。

##############################################################################
#                       运行应用
##############################################################################
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
