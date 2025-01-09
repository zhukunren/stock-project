import numpy as np
import pandas as pd
import os

# ---------- 技术指标计算函数 ----------

def compute_RSI(series, period=14):
    """
    RSI (Relative Strength Index) - 相对强弱指数
    衡量价格上涨和下跌的速度和幅度，用于判断超买或超卖状态。
    参数:
    - series: 序列 (如收盘价)
    - period: 计算周期
    返回:
    - RSI值序列
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(series, fast_period=12, slow_period=26, signal_period=9):
    """
    MACD (Moving Average Convergence Divergence) - 指数平滑异同移动平均线
    衡量短期和长期价格趋势之间的差异。
    参数:
    - series: 序列 (如收盘价)
    - fast_period: 快速均线周期
    - slow_period: 慢速均线周期
    - signal_period: 信号线周期
    返回:
    - MACD值, 信号线值
    """
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def compute_Bollinger_Bands(series, period=20, num_std=2):
    """
    Bollinger Bands - 布林带
    基于移动平均和标准差构造的价格波动区间。
    参数:
    - series: 序列 (如收盘价)
    - period: 移动平均周期
    - num_std: 标准差倍数
    返回:
    - 上轨, 中轨, 下轨
    """
    rolling_mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def compute_KD(high, low, close, period=14):
    """
    KD指标 (KDJ的基础)
    衡量当前价格相对于过去高点和低点的位置。
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - period: 计算周期
    返回:
    - K值, D值
    """
    low_min = low.rolling(window=period).min()
    high_max = high.rolling(window=period).max()
    rsv = (close - low_min) / (high_max - low_min) * 100
    K = rsv.ewm(com=2).mean()
    D = K.ewm(com=2).mean()
    return K, D

def compute_ATR(high, low, close, period=14):
    """
    ATR (Average True Range) - 平均真实波幅
    衡量价格波动范围的指标。
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - period: 计算周期
    返回:
    - ATR值序列
    """
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = hl.combine(hc, max).combine(lc, max)
    atr = tr.rolling(window=period).mean()
    return atr

def compute_ADX(high, low, close, period=14):
    """
    ADX (Average Directional Index) - 平均趋向指数
    衡量趋势强度的指标。
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - period: 计算周期
    返回:
    - +DI, -DI, ADX值
    """
    up_move = high.diff()
    down_move = low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * (-down_move)

    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = hl.combine(hc, max).combine(lc, max)
    tr_sum = tr.rolling(window=period).sum()

    plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr_sum)
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr_sum)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()
    return plus_di, minus_di, adx

def compute_CCI(high, low, close, period=20):
    """
    CCI (Commodity Channel Index) - 商品通道指标
    衡量价格偏离其统计均值的程度。
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - period: 计算周期
    返回:
    - CCI值序列
    """
    tp = (high + low + close) / 3
    ma = tp.rolling(window=period).mean()
    md = (tp - ma).abs().rolling(window=period).mean()
    cci = (tp - ma) / (0.015 * md)
    return cci

def compute_momentum(series, period=10):
    """
    Momentum - 动量指标
    衡量当前价格相对于过去N天价格的变化幅度，反映价格变化的速度和方向。
    参数:
    - series: 时间序列 (如收盘价)
    - period: 计算周期 (默认10)
    返回:
    - 动量值序列
    """
    return series.diff(period)

def compute_ROC(series, period=10):
    """
    ROC (Rate of Change) - 变化率指标
    衡量当前价格相对于过去N天价格的变化百分比，用于反映趋势的强弱。
    参数:
    - series: 时间序列 (如收盘价)
    - period: 计算周期 (默认10)
    返回:
    - ROC值序列（百分比）
    """
    return series.pct_change(period) * 100

def compute_volume_change(volume, period=10):
    """
    Volume Change - 成交量变化率
    衡量当前成交量相对于过去N天成交量的变化比例，用于捕捉市场活跃度的变化。
    参数:
    - volume: 成交量序列
    - period: 计算周期 (默认10)
    返回:
    - 成交量变化率序列
    """
    return volume.diff(period) / volume.shift(period)

def compute_VWAP(high, low, close, volume):
    """
    VWAP (Volume Weighted Average Price) - 成交量加权平均价
    衡量市场的平均成交成本，常用于判断价格的合理区间。
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - volume: 成交量序列
    返回:
    - VWAP值序列
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def compute_zscore(series, period=20):
    """
    Z-Score - 标准分数
    衡量当前值相对于过去N天均值的标准化偏差，反映价格的异常程度。
    参数:
    - series: 时间序列 (如收盘价)
    - period: 计算周期 (默认20)
    返回:
    - Z-Score值序列
    """
    mean = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return (series - mean) / std

def compute_volatility(series, period=10):
    """
    Volatility - 波动率
    衡量价格在过去N天的波动幅度，通常以标准差为度量。
    参数:
    - series: 时间序列 (如收盘价的收益率)
    - period: 计算周期 (默认10)
    返回:
    - 波动率序列
    """
    return series.pct_change().rolling(window=period).std()

def compute_OBV(close, volume):
    """
    OBV (On-Balance Volume) - 平衡成交量
    通过成交量的累积变化来衡量买卖力量，从而判断价格趋势的强弱。
    参数:
    - close: 收盘价序列
    - volume: 成交量序列
    返回:
    - OBV值序列
    用法:
    - OBV值随时间上升表示资金流入市场，可能预示价格上涨。
    - OBV值下降表示资金流出市场，可能预示价格下跌。
    """
    # 计算价格变化方向 (+1, 0, -1)
    direction = np.sign(close.diff())
    direction.iloc[0] = 0  # 第一天无法计算变化方向，设为0
    # 根据方向累积成交量
    obv = (volume * direction).fillna(0).cumsum()
    return obv

def compute_williams_r(high, low, close, period=14):
    """
    Williams %R - 威廉指标
    衡量当前收盘价相对于过去N天的高点和低点的位置，常用于超买和超卖状态的判断。
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - period: 计算周期 (默认14)
    返回:
    - Williams %R值序列
    用法:
    - %R接近-100: 表示超卖区域，可能出现反弹。
    - %R接近0: 表示超买区域，可能出现回调。
    """
    # 计算过去N天的最高点和最低点
    hh = high.rolling(window=period).max()
    ll = low.rolling(window=period).min()
    # 计算威廉指标
    wr = -100 * ((hh - close) / (hh - ll))
    return wr

def compute_MFI(high, low, close, volume, period=14):
    """
    MFI (Money Flow Index)
    类似于RSI，但考虑成交量。
    """
    tp = (high + low + close) / 3
    mf = tp * volume
    positive_flow = mf.where(tp > tp.shift(), 0)
    negative_flow = mf.where(tp < tp.shift(), 0)
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    mfi = 100 * (positive_sum / (positive_sum + negative_sum))
    return mfi

def compute_CMF(high, low, close, volume, period=20):
    """
    CMF (Chaikin Money Flow)
    衡量资金流入流出强度。
    """
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_volume = mf_multiplier * volume
    cmf = mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()
    return cmf

def compute_TRIX(series, period=15):
    """
    TRIX (Triple Exponential Average)
    衡量价格变化的速度，三重平滑的EMA变化率。
    """
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    trix = (ema3.diff() / ema3.shift()) * 100
    return trix

def compute_ultimate_oscillator(high, low, close, short_period=7, medium_period=14, long_period=28):
    """
    Ultimate Oscillator (UO)
    综合不同周期的摆动值衡量市场动能。
    """
    bp = close - np.minimum(low.shift(1), close.shift(1))
    tr = np.maximum(high - low, 
                    np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    avg7 = bp.rolling(short_period).sum() / tr.rolling(short_period).sum()
    avg14 = bp.rolling(medium_period).sum() / tr.rolling(medium_period).sum()
    avg28 = bp.rolling(long_period).sum() / tr.rolling(long_period).sum()

    uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / (4 + 2 + 1)
    return uo

def compute_chaikin_oscillator(high, low, close, volume, short_period=3, long_period=10):
    """
    Chaikin Oscillator
    基于ADL(累积/派发线)的MACD式指标。
    """
    adl = compute_ADL_line(high, low, close, volume)
    short_ema = adl.ewm(span=short_period, adjust=False).mean()
    long_ema = adl.ewm(span=long_period, adjust=False).mean()
    cho = short_ema - long_ema
    return cho

def compute_ADL_line(high, low, close, volume):
    """
    ADL (Accumulation/Distribution Line)
    """
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], np.nan).fillna(0)
    mf_volume = mf_multiplier * volume
    adl = mf_volume.cumsum()
    return adl

def compute_PPO(series, fast_period=12, slow_period=26):
    """
    PPO (Percentage Price Oscillator)
    与MACD类似，只是输出为百分比。
    """
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    ppo = (fast_ema - slow_ema) / slow_ema * 100
    return ppo

def compute_DPO(series, period=20):
    """
    DPO (Detrended Price Oscillator)
    去趋势价格振荡指标。
    """
    shifted = series.shift(int((period/2)+1))
    sma = series.rolling(window=period).mean()
    dpo = series - sma.shift(int((period/2)+1))
    return dpo

def compute_KST(series, r1=10, r2=15, r3=20, r4=30, sma1=10, sma2=10, sma3=10, sma4=15):
    """
    KST (Know Sure Thing)
    基于ROC的综合动量指标。
    """
    roc1 = series.pct_change(r1)*100
    roc2 = series.pct_change(r2)*100
    roc3 = series.pct_change(r3)*100
    roc4 = series.pct_change(r4)*100

    sma_roc1 = roc1.rolling(sma1).mean()
    sma_roc2 = roc2.rolling(sma2).mean()
    sma_roc3 = roc3.rolling(sma3).mean()
    sma_roc4 = roc4.rolling(sma4).mean()

    kst = sma_roc1 + 2*sma_roc2 + 3*sma_roc3 + 4*sma_roc4
    signal = kst.rolling(9).mean()
    return kst, signal

def compute_KAMA(series, n=10, pow1=2, pow2=30):
    """
    KAMA (Kaufman's Adaptive Moving Average)
    自适应移动平均
    """
    change = series.diff(n).abs()
    volatility = series.diff(1).abs().rolling(window=n).sum()
    er = change / volatility
    sc = (er * (2/(pow1+1)-2/(pow2+1)) + 2/(pow2+1))**2

    kama = series.copy()
    for i in range(n, len(series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i]*(series.iloc[i]-kama.iloc[i-1])
    return kama

# ---------- 高低点识别函数 ----------

def identify_high_peaks(df, window=3):
    df = df.copy()
    # 定义滚动窗口大小
    win = 2 * window + 1

    # 使用 NumPy 快速计算滚动最大值
    rolling_max = df['High'].rolling(window=win, center=True).max()

    # 标记潜在高点（等于滚动窗口最大值）
    df['PotentialPeak'] = (df['High'] == rolling_max).astype(int)

    # 计算窗口内最大值出现的次数
    # 使用 NumPy 的布尔操作替代 apply 函数
    rolling_max_counts = (
        df['High']
        .rolling(window=win, center=True)
        .apply(lambda x: np.sum(x == np.max(x)), raw=True)
    )

    # 标记最终的高点：既是潜在高点，又是窗口中唯一最大值
    df['Peak'] = ((df['PotentialPeak'] == 1) & (rolling_max_counts == 1)).astype(int)

    # 清理临时列
    df.drop(columns=['PotentialPeak'], inplace=True)

    return df


def identify_low_troughs(df, window=3):
    df = df.copy()
    # 定义滚动窗口大小
    win = 2 * window + 1

    # 使用 NumPy 快速计算滚动最小值
    rolling_min = df['Low'].rolling(window=win, center=True).min()

    # 标记潜在低点（等于滚动窗口最小值）
    df['PotentialTrough'] = (df['Low'] == rolling_min).astype(int)

    # 计算窗口内最小值出现的次数
    rolling_min_counts = (
        df['Low']
        .rolling(window=win, center=True)
        .apply(lambda x: np.sum(x == np.min(x)), raw=True)
    )

    # 标记最终的低点：既是潜在低点，又是窗口中唯一最小值
    df['Trough'] = ((df['PotentialTrough'] == 1) & (rolling_min_counts == 1)).astype(int)

    # 清理临时列
    df.drop(columns=['PotentialTrough'], inplace=True)

    return df



# ---------- 数据读取与处理函数 ----------

def read_day_fromtdx(file_path, stock_code_tdx):
    """
    从通达信DAY文件中读取股票日线数据。
    参数:
    - file_path: 文件目录路径
    - stock_code_tdx: 股票代码 (如 "sh600000")
    返回:
    - 包含日期、开高低收、成交量等列的DataFrame
    """
    file_full_path = os.path.join(file_path, 'vipdoc', stock_code_tdx[:2].lower(), 'lday', f"{stock_code_tdx}.day")
    print(f"尝试读取文件: {file_full_path}")
    dtype = np.dtype([
        ('date', '<i4'),
        ('open', '<i4'),
        ('high', '<i4'),
        ('low', '<i4'),
        ('close', '<i4'),
        ('amount', '<f4'),
        ('volume', '<i4'),
        ('reserved', '<i4')
    ])
    if not os.path.exists(file_full_path):
        print(f"文件 {file_full_path} 不存在。")
        return pd.DataFrame()
    try:
        data = np.fromfile(file_full_path, dtype=dtype)
        print(f"读取了 {len(data)} 条记录。")
    except Exception as e:
        print(f"读取文件失败：{e}")
        return pd.DataFrame()
    if data.size == 0:
        print("文件为空。")
        return pd.DataFrame()
    df = pd.DataFrame({
        'date': pd.to_datetime(data['date'].astype(str), format='%Y%m%d', errors='coerce'),
        'Open': data['open'] / 100.0,
        'High': data['high'] / 100.0,
        'Low': data['low'] / 100.0,
        'Close': data['close'] / 100.0,
        'Amount': data['amount'],
        'Volume': data['volume'],
    })
    df = df.dropna(subset=['date'])
    df['TradeDate'] = df['date'].dt.strftime('%Y%m%d')
    df.set_index('date', inplace=True)
    print(f"创建了包含 {len(df)} 条记录的DataFrame。")
    return df

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

'''
# 训练模型
def train_model_for_label(df, N, label_column, all_features, classifier_name, n_features_selected):
    # 针对特定的 label_column 训练模型
    print(f"开始训练 {label_column} 模型...")
    data = df.copy()
    # 特征与标签
    X = data[all_features]
    y = data[label_column]

    # 数据标准化
    print("标准化数据...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 处理类别不平衡（使用 SMOTE）
    print("使用 SMOTE 处理类别不平衡...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    print(f"SMOTE 后数据长度: {len(X_resampled)}")

    # 划分训练集和测试集
    print("划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    # 定义分类器
    clf_name, clf = get_classifier(classifier_name)

    # 使用 GridSearchCV 为分类器调参
    print(f"正在为分类器 {clf_name} 进行 GridSearchCV 调参...")
    param_grid = {}
    if clf_name == 'rf':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['auto', 'sqrt'],
            'bootstrap': [True, False]
        }
    elif clf_name == 'svc':
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    elif clf_name == 'lr':
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
    elif clf_name == 'gb':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
    else:
        raise ValueError(f"未知的分类器名称: {clf_name}")

    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='f1',
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    print(f"{clf_name} 的最佳参数: {grid_search.best_params_}")
    print(f"{clf_name} 的最佳得分: {grid_search.best_score_:.4f}")
    best_estimator = grid_search.best_estimator_

    # 特征选择（RFECV 或 RFE）
    print("开始特征选择...")
    if n_features_selected == 'auto':
        # 使用RFECV自动选择特征数量
        feature_selector = RFECV(estimator=best_estimator,
                                  step=max(1, len(all_features) // 10),
                                  cv=3,
                                  scoring='f1',
                                  min_features_to_select=10)
        feature_selector.fit(X_train, y_train)
        selected_features = [all_features[i] for i in range(len(all_features)) if feature_selector.support_[i]]
        print(f"RFECV选择的特征数量：{feature_selector.n_features_}")
    else:
        # 使用RFE手动选择特征数量
        feature_selector = RFE(estimator=best_estimator,
                               n_features_to_select=int(n_features_selected),
                               step=max(1, len(all_features) // 10))
        feature_selector.fit(X_train, y_train)
        selected_features = [all_features[i] for i in range(len(all_features)) if feature_selector.support_[i]]
        print(f"RFE选择的特征数量：{len(selected_features)}")

    print("选择的特征：")
    print(selected_features)

    # 使用选择的特征重新训练模型
    print("使用选择的特征重新训练模型...")
    X_train_selected = feature_selector.transform(X_train)
    X_test_selected = feature_selector.transform(X_test)

    best_estimator.fit(X_train_selected, y_train)

    # 模型预测
    print("进行模型预测...")
    y_pred = best_estimator.predict(X_test_selected)
    y_proba = best_estimator.predict_proba(X_test_selected)[:, 1]

    # 计算更多评价指标
    print("计算更多评价指标...")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # 模型评估
    print("模型评估：")
    print(classification_report(y_test, y_pred))
    print("混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))

    print(f"ROC AUC: {roc_auc:.2f}")
    print(f"PR AUC: {pr_auc:.2f}")
    print(f"精确率（Precision）：{precision:.2f}")
    print(f"召回率（Recall）：{recall:.2f}")
    print(f"Matthews相关系数（MCC）：{mcc:.2f}")

    # 特征重要性（仅适用于支持特征重要性的分类器，如随机森林、梯度提升）
    feature_importances = None
    if hasattr(best_estimator, 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'Feature': selected_features,
            'Importance': best_estimator.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print("特征重要性（前20个）：")
        print(feature_importances.head(20))
    elif hasattr(best_estimator, 'coef_'):
        feature_importances = pd.DataFrame({
            'Feature': selected_features,
            'Importance': best_estimator.coef_[0]
        }).sort_values(by='Importance', ascending=False)
        print("特征重要性（前20个）：")
        print(feature_importances.head(20))
    else:
        print("当前模型不支持特征重要性评估。")

    metrics = {
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc,
        'Precision': precision,
        'Recall': recall,
        'MCC': mcc
    }

    return best_estimator, scaler, feature_selector, selected_features, all_features, grid_search.best_score_, metrics
'''
'''
    base_features = [
        'Close_MA5_Diff', 'MA5_MA20_Diff', 'RSI_Signal', 'MACD_Diff',
        'Bollinger_Position', 'K_D_Diff', 'ConsecutiveUp', 'ConsecutiveDown',
        'Cross_MA5_Count', 'Volume_Spike_Count', 'one',
        'Price_MA20_Diff', 'ATR_14', 'Volatility_10', 'CCI_20','K_D_Diff', 'ConsecutiveUp', 'ConsecutiveDown']
   
    base_features.extend=['Williams_%R_14', 'OBV', 'VWAP',
        'ZScore_20', 'Plus_DI', 'Minus_DI', 'ADX_14',
        'Bollinger_Width', 'Slope_MA5', 'Volume_Change', 'Price_Mean_Diff', 'High_Mean_Diff', 'Low_Mean_Diff',
        'MA_5', 'MA_20', 'MA_50', 'MA_200', 'EMA_5', 'EMA_20'

    ]
    '''