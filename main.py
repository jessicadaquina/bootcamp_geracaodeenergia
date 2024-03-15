import pandas as pd
from matplotlib import pyplot as plt
from load_data import *
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import pmdarima as pmd 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# load all csv data
list = ["din_instante"]
df = load_data("data", list)
print(df)

df_test = load_data("test", list)

# Printing energy generation chart for the entire country
pd.pivot_table(df, index="din_instante", values="val_geracao").plot(kind="bar")
plt.legend(loc='best')

# Converting period date to timestamp
df_test["din_instante"] = df_test["din_instante"].dt.to_timestamp()

# Performing differentiation calculation
adf_test_result = adfuller(df['val_geracao'])
df['diff_load_data'] = df["val_geracao"].diff().fillna(0)
adf_test_result_diff = adfuller(df['diff_load_data'])

adf_test_result_diff = pd.DataFrame({
    'iTEM':[
         'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used',
        'Critical Value (1%)', 'Critical Value (5%)', 'Critical Value (10%)'
    ],
    'Value': [
        adf_test_result[0],  # Test statistic
        adf_test_result[1],  # p-value
        adf_test_result[2],  # #Lags Used
        adf_test_result[3],  # Number of Observations Used
        adf_test_result[4]['1%'],  # Critical Value for 1%
        adf_test_result[4]['5%'],  # Critical Value for 5%
        adf_test_result[4]['10%']  # Critical Value for 10%
    ]
})

print(adf_test_result_diff)

# Checking which model best fits the data
pdq_value =pmd.auto_arima(df['val_geracao'], start_p= 1, start_q= 1, test='adf', m= 12, seasonal= True, trace= True)

# Calculating Sarimax with database from 2000 to 2018
model = SARIMAX(df['val_geracao'], order=(4, 0, 5), seasonal_order=(1, 0, 1, 12))
model_fit = model.fit(disp=False)

# ACF AND PACF chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,6))

# autocorrelation chart
plot_acf(df['diff_load_data'], ax=ax1)
plot_pacf(df['diff_load_data'], ax=ax2, method='ywm')
plt.show()

# printing the statistics
print(model_fit.summary())

# Calculating Sarimax with database from 2019 to 2020
model1 = SARIMAX(df_test['val_geracao'], order=(4, 0, 5), seasonal_order=(1, 0, 1, 12))
model_fit1 = model1.fit(disp=False)

# Predictions
predictions = model_fit.get_prediction(start=0, end=len(df) - 1)
predicted_means = predictions.predicted_mean

# Assigning dates to a variable to plot actual data and forecasts together
predicted_dates = df['din_instante']

# printing historical data and forecasts
plt.figure(figsize=(15, 6))
df["din_instante"] = df["din_instante"].dt.to_timestamp()
plt.plot(df['din_instante'], df['val_geracao'], label='Dados reais', color='blue')
plt.plot(predicted_dates,predicted_means,label = 'Dados previstos',color ='red')
plt.title('Dados reais e Valores previstos para Geração de Energia')
plt.xlabel('Datas')
plt.ylabel('Valor de Geração')
plt.legend()
plt.show()

# Predictions of future dates
last_date = df['din_instante'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=24, freq='ME')

# Value predictions
forecast_object = model_fit.get_forecast(steps=24)

# Confidence interval
confianca = forecast_object.conf_int()

# Creating dataset with future values ​​collected
forecast_df = pd.DataFrame({
    'date': future_dates,
    'predicted_net_revenue': forecast_object.predicted_mean,
    'lower_confidence': confianca.iloc[:, 0],
    'upper_confidence': confianca.iloc[:, 1]
})

# Plotting historical data and forecasts
plt.figure(figsize=(15, 6))
plt.plot(df['din_instante'], df['val_geracao'], label='Dados Históricos', color='blue')
plt.plot(df_test['din_instante'], df_test['val_geracao'], label='Dados Futuros', color='green')
plt.plot(forecast_df['date'], forecast_df['predicted_net_revenue'], label='Predições', color='red', linestyle='--')
plt.title('Geração de energia mensal')
plt.xlabel('Mês/Ano')
plt.ylabel('Valor de Geração')
plt.legend()
plt.show()


# Calculating R², MAE, MSE, RMSE
y_true = df_test['val_geracao']
y_pred = forecast_df['predicted_net_revenue']

# R²
r2 = r2_score(y_true, y_pred)

# MAE
mae = mean_absolute_error(y_true, y_pred)

# MSE
mse = mean_squared_error(y_true, y_pred)

# RMSE
rmse = mean_squared_error(y_true, y_pred, squared=False)

print("R²:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Generating energy generation graph in Brazil and Paraguay
list = ["din_instante"]
list_id_estado = ["din_instante","id_estado"]

df = load_data("data", list_id_estado)
df_test= load_data("test", list_id_estado)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,6))
pd.pivot_table(df, index=["id_estado"], values="val_geracao").plot(kind="bar", ax=ax1)
pd.pivot_table(df_test, index=["id_estado"], values="val_geracao").plot(kind="bar", ax = ax2)
plt.legend(loc='best')
plt.show()

print("end")