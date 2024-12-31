# Prediksi-Medan-magnet-
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import matplotlib.dates as mdates
from google.colab import files

# Upload Excel file
print("Silakan unggah file Excel Anda.")
uploaded = files.upload()

# Read the uploaded Excel file
file_name = list(uploaded.keys())[0]  # Get the uploaded file name
data = pd.read_excel(file_name)

# Select relevant columns and rename them for Prophet
data = data[['Date', 'F (nT)']].rename(columns={'Date': 'ds', 'F (nT)': 'y'})
data['ds'] = pd.to_datetime(data['ds'], unit='m')  # Convert minutes to datetime

# Remove duplicates and sort data by time
data = data.drop_duplicates(subset=['ds']).sort_values(by='ds')

# Aggregate data every 5 minutes to reduce granularity
data = data.set_index('ds').resample('5T').mean().reset_index()

# Display dataset after preprocessing
print("Dataset setelah preprocessing:")
print(data.head())

# Initialize Prophet model with optimized parameters
model = Prophet(
    growth='linear',
    changepoint_prior_scale=0.05,  # Reduce sensitivity to trend changes to avoid overfitting
    seasonality_prior_scale=10,  # Increase sensitivity to seasonality
    holidays_prior_scale=10,  # Reduce holiday effects if related data is available
    daily_seasonality=True,  # Add daily seasonality
    yearly_seasonality=False,  # Disable yearly seasonality, depending on the dataset
    weekly_seasonality=True  # Enable weekly seasonality for longer-term predictions
)

# Add additional custom seasonality (if needed)
model.add_seasonality(name='custom', period=30, fourier_order=5)

# Fit the model
model.fit(data)

# Predict for the next couple of days (e.g., 2 days = 2 * 24 * 60 minutes)
days_ahead = 2
future = model.make_future_dataframe(periods=days_ahead * 24 * 60, freq='T')  # 'T' for minutes
forecast = model.predict(future)

# Add actual data to predictions for comparison
forecast['actual'] = None
forecast.loc[:len(data)-1, 'actual'] = data['y'].values

# Calculate accuracy metrics
# Filter historical data where both actual and predicted values are available
historical = forecast.loc[:len(data)-1]
actual = historical['actual']
predicted = historical['yhat']

# Calculate metrics
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)

print(f"Accuracy Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Plot predictions vs actual values
plt.figure(figsize=(14, 7))
plt.plot(data['ds'], data['y'], 'bo-', label='Data Aktual (F)')
plt.plot(forecast['ds'], forecast['yhat'], 'orange', label='Prediksi (F)')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                 color='orange', alpha=0.2, label='Uncertainty Interval')

# Ubah tampilan waktu menjadi tiap 4 jam
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)

plt.title(f"Prediksi Magnetic Field (F) untuk {days_ahead} Hari ke Depan")
plt.xlabel("Waktu")
plt.ylabel("F (nT)")
plt.legend()
plt.grid()
plt.show()

# Plot residuals (errors)
residuals = actual - predicted
plt.figure(figsize=(14, 7))
plt.plot(historical['ds'], residuals, label='Residuals', color='red')
plt.axhline(0, linestyle='--', color='black', alpha=0.7)

# Ubah tampilan waktu menjadi tiap 4 jam
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)

plt.title("Residuals Over Time")
plt.xlabel("Waktu")
plt.ylabel("Residual (Actual - Predicted)")
plt.legend()
plt.grid()
plt.show()

# Plot residuals histogram
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color='purple', alpha=0.7, edgecolor='black')
plt.title("Histogram of Residuals")
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.grid(axis='y')
plt.show()

# Plot bar chart for comparison of actual and predicted data
plt.figure(figsize=(14, 7))
# Actual data
plt.bar(forecast['ds'][:len(data)], forecast['actual'][:len(data)], color='blue', label='Data Aktual', width=0.4, align='center')
# Predicted data
plt.bar(forecast['ds'][len(data):], forecast['yhat'][len(data):], color='orange', alpha=0.7, label='Prediksi', width=0.4, align='center')

# Ubah tampilan waktu menjadi tiap 4 jam
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)

plt.title(f"Perbandingan Data Aktual dan Prediksi Magnetic Field (F) ({days_ahead} Hari)")
plt.xlabel("Waktu")
plt.ylabel("F (nT)")
plt.legend()
plt.grid(axis='y')
plt.show()
