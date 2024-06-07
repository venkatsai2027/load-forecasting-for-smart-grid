import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('ANN_data.csv')
data = df['BSES RAJDHANI'].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

sequence_length = 12  # You can adjust this based on your requirements
sequences = []
targets = []
for i in range(len(data_scaled) - sequence_length):
    seq = data_scaled[i:i+sequence_length]
    target = data_scaled[i+sequence_length:i+sequence_length+1]
    sequences.append(seq)
    targets.append(target)

X = np.array(sequences)
y = np.array(targets)

split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model = Sequential()
model.add(GRU(units=50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions to the original scale
predictions_original_scale = scaler.inverse_transform(predictions)
actual_values_original_scale = scaler.inverse_transform(y_test.reshape(-1,1))

import matplotlib.pyplot as plt

# Inverse transform the actual values to the original scale
# actual_values_original_scale = scaler.inverse_transform(y_test)

# Plotting the actual values vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(actual_values_original_scale, label='Actual Load Demand', color='blue')
plt.plot(predictions_original_scale, label='Predicted Load Demand', color='red')
# plt.plot(actual_values_original_scale, label='Actual Values', linestyle='-', marker='o', color='b')
# plt.plot(predictions_original_scale, label='Predicted Values', linestyle='-', marker='X', color='r')
plt.title('Actual vs Predicted Load Demand(BSES RAJDHANI)')
plt.xlabel('Time')
plt.ylabel('Load Demand')
plt.legend()
plt.show()


import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame containing the "DATE/TIME" column
# Assuming 'predictions_original_scale' is a numpy array of predicted values
# Assuming 'y_test' is a numpy array of actual values
# Assuming 'split_index' is the index where you split your data into training and testing sets

# Extract the "DATE/TIME" column
dates = df.index[split_index + sequence_length:]  # Assuming the date is the index of your DataFrame

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(dates, actual_values_original_scale, label='Actual', color='blue')
plt.plot(dates, predictions_original_scale, label='Predicted', color='red' )
plt.xlabel('Date/Time')
plt.ylabel('Load Demand')
plt.title('Actual vs Predicted Load Demand(BSES RAJDHANI)')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Inverse transform the actual values to the original scale
# actual_values_original_scale = scaler.inverse_transform(y_test)

# Plotting the actual values vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(actual_values_original_scale[-200:], label='Actual Load Demand', color='blue')
plt.plot(predictions_original_scale[-200:], label='Predicted Load Demand', color='red')
plt.title('Actual vs Predicted Load Demand(BSES RAJDHANI)')
plt.xlabel('Time')
plt.ylabel('Load Demand')
plt.legend()