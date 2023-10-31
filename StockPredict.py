# %% [markdown]
# # LSTM Stock Prediction Model
# By Joshua Jenkins

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.utils import plot_model
import yfinance as yf

# %%
#Get the Dataset
ticker = input("Enter ticker: ")
start_date = input("Enter start date in format YYYY-MM-DD: ")
end_date = date.today().strftime("%Y-%m-%d")
data = yf.download(ticker, start=start_date, end=end_date)

# Save the data to a CSV file
data.to_csv("Ticker.csv")
df=pd.read_csv("Ticker.csv",na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=True)
df["Date"] = pd.to_datetime(df.index,format='%Y-%m-%d')
df.head()

# %%
#Print the shape of Dataframe and check for null values
print("Dataframe Shape:",df.shape)
print("Null Values (if any):",df.isnull().values.any())


# %%
#Plot the closing price of the stock
df.plot(x='Date',y='Adj Close', xlabel = 'Date', ylabel = 'Price (USD)', title = 'Microsoft Stock Price')

# %%
#Set target variable as the closing price
output_var = pd.DataFrame(df['Adj Close'])
#Selecting the features
features = ['Open','High','Low','Volume']

# %%
#Setting up scaler
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform = pd.DataFrame(data=feature_transform, columns=features, index=df.index)
feature_transform.head()
feature_transform

# %%
#Splitting to train and test set
timesplit = TimeSeriesSplit(n_splits=10)#?
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
    Y_train, Y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()

# %%
#Data Proessing for LSTM
trainX = np.array(X_train)
testX = np.array(X_test)
X_train = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
X_test = testX.reshape(testX.shape[0], 1, testX.shape[1])


# %%
#LSTM Model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
plot_model(lstm, show_shapes = True, show_layer_names =True)

# %%
#Training the Model
lstm.fit(X_train, Y_train, epochs=200, batch_size=8, verbose=1, shuffle=False)

# %%
#Prediction
Y_pred = lstm.predict(X_test)
Y_pred

# %%
# Plotting the values up until today
plt.plot(df.iloc[test_index]['Date'], Y_test, label='True Value')
plt.plot(df.iloc[test_index]['Date'], Y_pred, label='Predicted Value')
plt.title('Stock Price Prediction vs. True Value')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# %%
# Fetch the latest data and making a future prediction
latest_data = df.iloc[-1][features].values.reshape(1, len(features))
latest_data_scaled = scaler.transform(latest_data.reshape(1, -1))
next_day_prediction = lstm.predict(latest_data_scaled.reshape(1, 1, len(features)))
next_day = df.index[-1] + pd.Timedelta(days=1)
print("Predicted Stock Price for " + next_day.strftime('%Y-%m-%d') + ":", next_day_prediction[0, 0])

# %%
# Filter the dataframe to include only the last week of data
last_week_df = df.loc[df.index >= df.index[-1] - pd.DateOffset(weeks=1)]
last_week_df['Date'] = last_week_df.index.strftime('%m-%d')

# Plot the predicted value along with the previous values for the last week
plt.plot(last_week_df['Date'], last_week_df['Adj Close'], label='Historical Values Last Week')
plt.scatter(next_day.strftime('%m-%d'), next_day_prediction, color='g', label='Predicted Next Day Value')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Stock Price Prediction for the Last Week")
plt.legend()
plt.show()
