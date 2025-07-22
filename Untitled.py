#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[1]:  读取数据
import pandas as pd
data_original_1 = pd.read_excel('您的数据集路径',
                sheet_name=0, 
                parse_dates=['ds'],  # 将日期解析为 datetime 类型，
                index_col='ds',  # 将日期列设置为索引
                thousands=',')  #处理数字中的千分位符号，将其转换为数字
#print(data_original_1.info())  # 显示数据基本信息
#print(data_original_1.tail(10))  #查看最后10行

# In[2]:  STL分解
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# STL参数完整配置
stl = STL(
    endog = data_original_1['y'], 
    period = 365,  
    robust = True  # 是否使用鲁棒版本，抗异常值影响，推荐启用
)

# 拟合STL模型
result = stl.fit()

# 提取分解结果
trend_1 = result.trend
seasonal_1 = result.seasonal
resid_1 = result.resid

# 可视化三分量
plt.figure(figsize=(20, 10))
plt.subplot(3, 1, 1)
plt.plot(data_original_1.index, trend_1, color='blue', label='Trend')
plt.title('Trend Component')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(data_original_1.index, seasonal_1, color='green', label='Seasonal')
plt.title('Seasonal Component')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(data_original_1.index, resid_1, color='red', label='Residual')
plt.title('Residual Component')
plt.grid(True)
plt.legend()

plt.suptitle('STL Decomposition of Time Series', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# In[3]:  检验STL分解情况
# import numpy as np
# from statsmodels.stats.diagnostic import acorr_ljungbox

# # 1. 残差方差 & 标准差
# resid_var = np.var(resid)
# resid_std = np.std(resid)

# # 2. STL解释能力（类似R²）
# r2_stl = 1 - resid_var / np.var(data_original_1['y'].values)

# # 3. 各分量方差占比
# trend_ratio = np.var(trend_1) / np.var(data_original_1['y'].values)
# seasonal_ratio = np.var(seasonal_1) / np.var(data_original_1['y'].values)
# resid_ratio = resid_var / np.var(data_original_1['y'].values)

# # 4. Ljung-Box 检验（残差是否为白噪声）
# ljungbox_result = acorr_ljungbox(resid, lags=[10], return_df=True)
# ljungbox_pvalue = ljungbox_result['lb_pvalue'].values[0]

# # 输出评估结果
# print("\n========== STL 分解效果评估 ==========")
# print(f"残差方差: {resid_var:.4f}")
# print(f"残差标准差: {resid_std:.4f}")
# print(f"解释率 R²: {r2_stl:.4f}")
# print(f"趋势方差占比: {trend_ratio:.4f}")
# print(f"季节方差占比: {seasonal_ratio:.4f}")
# print(f"残差方差占比: {resid_ratio:.4f}")
# print(f"Ljung-Box p 值（残差白噪声检验）: {ljungbox_pvalue:.4f}")
# if ljungbox_pvalue > 0.05:
#     print("残差接近白噪声，分解效果较好。")
# else:
#     print("残差存在周期性，可能未充分分解。")
# print("======================================\n")

# In[4]:  划分训练集、测试集（也可以选择划分训练集、验证集和测试集）
# ========== 划分比例参数 ==========
train_ratio = 0.8    
test_ratio = 0.2      

total_len = len(data_original_1)
train_size_1 = int(train_ratio * total_len)
test_start_idx_1 = int((1 - test_ratio) * total_len)
# ========== 原始序列划分==========
train_data_1 =data_original_1[['y']][:train_size_1]
test_data_1 = data_original_1[['y']][test_start_idx_1:]
# ========== 趋势分量 ==========
trend_data_1 = pd.DataFrame({'y': trend_1})
train_trend_1 = trend_data_1[:train_size_1]
test_trend_1 = trend_data_1[test_start_idx_1:]
# ========== 季节性分量==========
seasonal_data_1 = pd.DataFrame({
    'ds': data_original_1.index,  # Prophet需要日期列
    'y': seasonal_1
})
train_seasonal_1 = seasonal_data_1[:train_size_1]
test_seasonal_1 = seasonal_data_1[test_start_idx_1:]
# ========== 残差分量 ==========
resid_data_1 = pd.DataFrame({'y': resid_1})
train_resid_1 = resid_data_1[:train_size_1]
test_resid_1 = resid_data_1[test_start_idx_1:]

# In[8]:  检验序列平稳性
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pandas as pd

def adf_test(series, name="序列"):
    result = adfuller(series.dropna())
    adf_stat, p_value, usedlag, nobs, crit_values, icbest = result
    print(f"\n【{name}】ADF检验结果：")
    print(f"  ADF统计量     : {adf_stat:.4f}")
    print(f"  p值           : {p_value:.4f}")
    for k, v in crit_values.items():
        print(f"  {k}% 临界值   : {v:.4f}")
    if p_value < 0.05:
        print("  结论：序列平稳（无需差分）")
    else:
        print("  结论：序列非平稳（需进一步差分）")
    return p_value < 0.05

def analyze_differencing(df, col="y", lags=40):
    df = df.copy()
    df["diff_1"] = df[col].diff(1)
    df["diff_2"] = df["diff_1"].diff(1)

    plt.figure(figsize=(18, 8))
    plt.subplot(3, 1, 1)
    plt.plot(df[col], label="原始序列")
    plt.title("原始序列")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(df["diff_1"], label="一阶差分")
    plt.title("一阶差分")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(df["diff_2"], label="二阶差分")
    plt.title("二阶差分")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ADF 检验
    adf_test(df[col], "原始序列")
    adf_test(df["diff_1"], "一阶差分")
    adf_test(df["diff_2"], "二阶差分")

    return df

diff_df = analyze_differencing(train_trend_1, col="y", lags=40)

# In[9]:  绘制ACF/PACF图

def plot_acf_pacf_for_diff(df, diff_col="diff_1", lags=40):
    df = df.dropna()

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    plot_acf(df[diff_col], ax=axes[0], lags=lags)
    axes[0].set_title(f"{diff_col} 的 ACF")
    plot_pacf(df[diff_col], ax=axes[1], lags=lags)
    axes[1].set_title(f"{diff_col} 的 PACF")
    plt.tight_layout()
    plt.show()

# 根据结果选择使用哪一阶差分：如果一阶差分平稳，使用 diff_1 去绘制 ACF / PACF，选择 pq 参数
plot_acf_pacf_for_diff(diff_df, diff_col="diff_1", lags=40)

# In[10]: BIC判断p和q

# #定阶，采用BIC贝叶斯信息准则判断p、q
# from statsmodels.tsa.arima.model import ARIMA
# import numpy as np

# # 最大 p 和 q 的范围
# pmax = 10
# qmax = 10

# # 初始化 BIC 矩阵，使用 np.nan 处理无效值
# bic_matrix = np.full((pmax + 1, qmax + 1), np.nan)

# # 遍历 p 和 q 并计算 BIC 值
# for p in range(pmax + 1):
#     for q in range(qmax + 1):
#         try:
#             model = ARIMA(train_trend["y"], order=(p, 1, q))
#             results = model.fit()
#             bic_matrix[p, q] = results.bic  # 存储 BIC 值
#         except Exception as e:
#             print(f"ARIMA fitting failed for p={p}, q={q}: {e}")

# # 将 BIC 矩阵转换为 Pandas DataFrame
# bic_df = pd.DataFrame(bic_matrix, columns=range(qmax + 1), index=range(pmax + 1))

# # 找到 BIC 值最小的 p 和 q
# min_idx = np.unravel_index(np.nanargmin(bic_matrix), bic_matrix.shape)
# p, q = min_idx

# print(f'BIC最小的p值和q值为：{p}、{q}')


# In[17]: 训练ARIMA模型

from statsmodels.tsa.arima.model import ARIMA
# ========== 模型参数设定==========
p, d, q = 2,0,8 
# ========== 构建并训练ARIMA模型 ==========
# 拟合趋势分量的ARIMA模型
train_series_arima_1 = train_trend_1['y'].dropna()

try:
    arima_model_1 = ARIMA(train_series_arima_1, order=(p, d, q))
    arima_result_1 = arima_model_1.fit()
    print("ARIMA模型训练成功，模型摘要如下：")
    print(arima_result_1.summary())
except Exception as e:
    print("ARIMA 模型训练失败，错误信息如下：")
    print(e)

# In[18]: 趋势分量预测

# ========== 训练集拟合值 ==========
train_trend_forecast_1 = arima_result_1.fittedvalues  # 已拟合的趋势预测值
train_trend_forecast_1.name = "ARIMA拟合"

# ========== 测试集预测 ==========
# 预测未来 len(test_trend_1) 步趋势值
steps = len(test_trend_1)
test_trend_forecast_1 = arima_result_1.forecast(steps=steps)

# 对齐索引
test_trend_forecast_1.index = test_trend_1.index
test_trend_forecast_1.name = "ARIMA预测"

# In[21]:  绘图和误差计算

from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
# ========== 可视化：训练集趋势拟合 ==========
plt.figure(figsize=(20, 6))
plt.plot(train_trend_1.index, train_trend_1['y'], label='真实值', color='black')
plt.plot(train_trend_forecast_1.index, train_trend_forecast_1, label='预测值', color='red', linestyle='--')
plt.title("训练集趋势分量拟合结果", fontproperties=font_prop)
plt.xlabel("日期", fontproperties=font_prop)
plt.ylabel("数值", fontproperties=font_prop)
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(prop=font_prop)
plt.tight_layout()
plt.show()

# ========== 可视化：测试集趋势预测 ==========
plt.figure(figsize=(20, 6))
plt.plot(test_trend_1.index, test_trend_1['y'], label='真实值', color='blue')
plt.plot(test_trend_forecast_1.index, test_trend_forecast_1, label='预测值', color='red', linestyle='--')
plt.title("测试集趋势分量预测结果", fontproperties=font_prop)
plt.xlabel("日期", fontproperties=font_prop)
plt.ylabel("数值", fontproperties=font_prop)
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(prop=font_prop)
plt.tight_layout()
plt.show()

# ========== 误差评估 ==========
train_rmse_a = np.sqrt(mean_squared_error(train_trend_1['y'], train_trend_forecast_1))
train_mae_a = mean_absolute_error(train_trend_1['y'], train_trend_forecast_1)
test_rmse_a = np.sqrt(mean_squared_error(test_trend_1['y'], test_trend_forecast_1))
test_mae_a = mean_absolute_error(test_trend_1['y'], test_trend_forecast_1)

print("\n[趋势分量 - 误差评估]")
print(f"训练集 RMSE: {train_rmse_a:.4f}")
print(f"训练集 MAE : {train_mae_a:.4f}")
print(f"测试集 RMSE: {test_rmse_a:.4f}")
print(f"测试集 MAE : {test_mae_a:.4f}")


# In[23]:  构建Prophet模型

from prophet import Prophet
import pandas as pd

#自定义特殊时期
def create_holiday_df(holiday_name, start_date, end_date):
    return pd.DataFrame({
        'holiday': holiday_name,
        'ds': pd.to_datetime([start_date]),
        'lower_window': -30,
        'upper_window': (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 30
    })
holidays_winter = pd.concat([
    create_holiday_df('winter_break_2021', '2021-01-09', '2021-02-28'),
    create_holiday_df('winter_break_2022', '2022-01-10', '2022-02-27'),
    create_holiday_df('winter_break_2023', '2023-01-09', '2023-02-28'),
    create_holiday_df('winter_break_2024', '2024-01-15', '2024-02-24')
])
holidays_summer = pd.concat([
    create_holiday_df('summer_break_2020', '2020-07-04', '2020-09-05'),
    create_holiday_df('summer_break_2021', '2021-07-10', '2021-09-05'),
    create_holiday_df('summer_break_2022', '2022-07-11', '2022-08-28'),
    create_holiday_df('summer_break_2023', '2023-07-01', '2023-09-03'),
    create_holiday_df('summer_break_2024', '2024-07-01', '2024-09-03')
])
promotion_days  = pd.concat([holidays_winter, holidays_summer]) 

# 构建 Prophet 模型
prophet_model_1 = Prophet(
    growth='linear',
    changepoint_range=0.8,
    changepoint_prior_scale=0.05,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    seasonality_prior_scale=15.0,
    holidays_prior_scale=15.0,
    interval_width=0.85,
    holidays=promotion_days 
)

# 添加节假日
prophet_model_1.add_country_holidays(country_name='CN')

# 拟合模型
prophet_model_1.fit(train_seasonal_1)

# In[25]:  季节分量预测

train_ds = train_seasonal_1.index.to_frame(index=False)
test_ds = test_seasonal_1.index.to_frame(index=False)

prophet_model_1.make_future_dataframe(periods=len(test_seasonal_1), freq='D')

# 模型预测
prophet_forecast_1 = prophet_model_1.predict(prophet_model_1)

# 提取所需字段并设置时间索引
prophet_forecast_data_1 = prophet_forecast_1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
prophet_forecast_data_1.set_index('ds', inplace=True)

# 区分训练集和测试集预测值
train_seasonal_forecast_1 = prophet_forecast_data_1.iloc[:train_size_1][['yhat']].rename(columns={'yhat': '预测值'})
test_seasonal_forecast_1 = prophet_forecast_data_1.iloc[test_start_idx_1:][['yhat']].rename(columns={'yhat': '预测值'})


# In[27]:  绘图和计算误差
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# ========== 训练集对比图 ==========
plt.figure(figsize=(18, 6))
plt.plot(train_seasonal_1.index, train_seasonal_1['y'], label='真实值', color='blue')
plt.plot(train_seasonal_forecast_1.index, train_seasonal_forecast_1['预测值'], label='拟合值', color='red', linestyle='--')
plt.title("季节分量训练集拟合结果", fontsize=16)
plt.xlabel("日期", fontsize=12)
plt.ylabel("数值", fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 测试集对比图 ==========
plt.figure(figsize=(18, 6))
plt.plot(test_seasonal_1.index, test_seasonal_1['y'], label='真实值', color='blue')
plt.plot(test_seasonal_1.index, test_seasonal_forecast_1['预测值'], label='预测值', color='red', linestyle='--')
plt.title("季节分量测试集预测结果", fontsize=16)
plt.xlabel("日期", fontsize=12)
plt.ylabel("数值", fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 误差计算 ==========
train_rmse_p = np.sqrt(mean_squared_error(train_seasonal_1['y'], train_seasonal_forecast_1['预测值']))
train_mae_p = mean_absolute_error(train_seasonal_1['y'], train_seasonal_forecast_1['预测值'])
test_rmse_p = np.sqrt(mean_squared_error(test_seasonal_1['y'], test_seasonal_forecast_1['预测值']))
test_mae_p = mean_absolute_error(test_seasonal_1['y'], test_seasonal_forecast_1['预测值'])
print("Prophet季节分量预测误差评估")
print(f"训练集：RMSE = {train_rmse_p:.4f}，MAE = {train_mae_p:.4f}")
print(f"测试集：RMSE = {test_rmse_p:.4f}，MAE = {test_mae_p:.4f}")

# In[29]:  构建LSTM模型
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.font_manager as font_manager

# ====== 参数设置 ======
look_back = 14
forecast_horizon = 7
dropout_rate = 0.3
epochs = 100
batch_size = 64

# ====== 数据缩放 ======
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_resid_1.values.reshape(-1, 1))
test_scaled = scaler.transform(test_resid_1.values.reshape(-1, 1))

# ====== 构造多步预测数据集 ======
def create_multistep_dataset(data, look_back, forecast_horizon):
    X, Y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:i + look_back, 0])
        Y.append(data[i + look_back:i + look_back + forecast_horizon, 0])
    return np.array(X), np.array(Y)

X, Y = create_multistep_dataset(train_scaled, look_back, forecast_horizon)
X = X.reshape((X.shape[0], look_back, 1))
Y = Y.reshape((Y.shape[0], forecast_horizon))

# ====== 构建 LSTM 模型 ======
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(dropout_rate),
    LSTM(48),
    Dropout(dropout_rate),
    Dense(forecast_horizon)
])
lstm_model.compile(optimizer='adam', loss='mse')

# ====== 模型训练 ======
history = lstm_model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=2)

# ====== 可视化训练过程 ======
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.title('LSTM Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# In[30]:  残差分量预测
# ====== 滑动窗口多步预测函数 ======
def sliding_window_forecast(model, init_seq, steps, look_back, forecast_horizon):
    result = []
    input_seq = init_seq.copy()
    while len(result) < steps:
        pred = model.predict(input_seq.reshape(1, look_back, 1), verbose=0)[0]
        n_output = min(forecast_horizon, steps - len(result))
        result.extend(pred[:n_output])
        input_seq = np.roll(input_seq, -n_output)
        input_seq[-n_output:] = pred[:n_output]
    return np.array(result)

# ====== 测试集预测 ======
init_seq = train_scaled[-look_back:, 0]
test_len = len(test_resid_1)
test_pred_scaled = sliding_window_forecast(lstm_model, init_seq, test_len, look_back, forecast_horizon)
test_resid_forecast_1 = scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()

# ====== 训练集拟合 ======
train_pred_scaled = lstm_model.predict(X, verbose=0)
train_pred_firststep = train_pred_scaled[:, 0]
train_resid_forecast_1 = np.full(len(train_resid_1), np.nan)
pred_vals_train = scaler.inverse_transform(train_pred_firststep.reshape(-1, 1)).flatten()
train_resid_forecast_1[-len(pred_vals_train):] = pred_vals_train

# ====== 可视化函数 ======
def plot_forecast(true_values, pred_values, index, title, label_true, label_pred):
    plt.figure(figsize=(18, 8))
    plt.plot(index, true_values, label=label_true, color='blue', linewidth=2)
    plt.plot(index, pred_values, label=label_pred, color='red', linestyle='--', linewidth=2)
    plt.title(title, fontproperties=font_prop)
    plt.xlabel("日期", fontproperties=font_prop)
    plt.ylabel("数值", fontproperties=font_prop)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.legend(prop=font_prop)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ====== 绘图展示 ======
plot_forecast(train_resid_1.values, train_resid_forecast_1 , train_resid_1.index,
              "LSTM 模型训练集拟合结果", "训练集真实值", "训练集拟合值（LSTM）")
plot_forecast(test_resid_1.values, test_resid_forecast_1, test_resid_1.index,
              "LSTM 模型测试集预测结果", "测试集真实值", "测试集预测值（LSTM）")

# ====== 误差指标计算 ======
train_rmse_1 = np.sqrt(mean_squared_error(train_resid_1.values, train_resid_forecast_1))
train_mae_1 = mean_absolute_error(train_resid_1.values, train_resid_forecast_1)
test_rmse_1 = np.sqrt(mean_squared_error(test_resid_1.values, test_resid_forecast_1))
test_mae_1 = mean_absolute_error(test_resid_1.values, test_resid_forecast_1)
print(f"LSTM 模型训练集RMSE: {train_rmse_1:.4f}, MAE: {train_mae_1:.4f}")
print(f"LSTM 模型测试集RMSE: {test_rmse_1:.4f}, MAE: {test_mae_1:.4f}")

# In[33]:  分量重构、绘图和计算误差

# ======构建完整预测值 ======
train_forecast_1= (
    train_trend_forecast_1.values.flatten() +
    train_seasonal_forecast_1['预测值'].values.flatten() +
    train_resid_forecast_1
)
train_forecast_1 = np.maximum(train_forecast_1, 0)  # 负值归零
train_forecast_1 = pd.Series(train_forecast_1, index=train_resid_1.index) 
test_forecast_1 = (
    test_trend_forecast_1.values.flatten() +
    test_seasonal_forecast_1['预测值'].values.flatten() +
    test_resid_forecast_1
)
test_forecast_1 = np.maximum(test_forecast_1, 0)  # 负值归零
test_forecast_1 = pd.Series(test_forecast_1, index=test_resid_1.index)  

# ====== 绘图对比预测与真实值 ======
plt.figure(figsize=(20, 8))
plt.plot(train_resid_1.index, data_original_1.loc[train_resid_1.index, 'y'], label='训练集真实值', color='blue')
plt.plot(train_forecast_1.index, train_forecast_1, label='训练集拟合值（组合模型）', color='red', linestyle='-')
plt.title("原始数据训练集拟合值与真实值对比", fontsize=16)
plt.xlabel("日期", fontsize=12)
plt.ylabel("数值", fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 8))
plt.plot(test_resid_1.index, data_original_1.loc[test_resid_1.index, 'y'], label='测试集真实值', color='blue')
plt.plot(test_forecast_1.index, test_forecast_1, label='测试集预测值（组合模型）', color='red', linestyle='-')
plt.title("原始数据测试集预测值与真实值对比", fontsize=16)
plt.xlabel("日期", fontsize=12)
plt.ylabel("数值", fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====== 误差评估 ======
train_rmse = np.sqrt(mean_squared_error(train_data_1, train_forecast_1))
train_mae = mean_absolute_error(train_data_1, train_forecast_1)
test_rmse = np.sqrt(mean_squared_error(test_data_1, test_forecast_1))
test_mae = mean_absolute_error(test_data_1, test_forecast_1)
print(f"训练集 RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
print(f"测试集 RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

