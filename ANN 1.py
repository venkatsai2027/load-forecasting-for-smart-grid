import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

data = pd.read_csv("ANN_data.csv")
# Convert 'DATE/TIME' column to datetime
data['DATE/TIME'] = pd.to_datetime(data['DATE/TIME'])
# Sort the data by date/time
data.sort_values(by='DATE/TIME', inplace=True)
# Extract 'BSES' (load values) column
load_values = data['BSES RAJDHANI'].values
# Normalize the load values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_load_values = scaler.fit_transform(load_values.reshape(-1, 1))

#Define sequence length (number of historical load values to consider)
sequence_length = 10
# Create input and output sequences
X, y = [], []
for i in range(len(scaled_load_values) - sequence_length):
    X.append(scaled_load_values[i:i+sequence_length].flatten())
    y.append(scaled_load_values[i+sequence_length][0])
X, y = np.array(X), np.array(y)

split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_ann_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping, model_checkpoint])

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Load the best model
model.load_weights('best_ann_model.h5')
# Make predictions
predictions = model.predict(X_test)
# Inverse transform predictions
predicted_load_values = scaler.inverse_transform(predictions)

actual_values_original_scale = scaler.inverse_transform(y_test.reshape(-1,1))


import matplotlib.pyplot as plt

# Inverse transform the actual values to the original scale
# actual_values_original_scale = scaler.inverse_transform(y_test)

# Plotting the actual values vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(actual_values_original_scale[-200:], label='Actual Load Demand', color='blue')
plt.plot(predicted_load_values[-200:], label='Predicted Load Demand', color='red')
plt.title('Actual vs. Predicted Load Demand(BSES RAJDHANI)')
plt.xlabel('Time')
plt.ylabel('Load Demand')
plt.legend()

import matplotlib.pyplot as plt

# Inverse transform the actual values to the original scale
# actual_values_original_scale = scaler.inverse_transform(y_test)

# Plotting the actual values vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(actual_values_original_scale, label='Actual Load Demand', color='blue')
plt.plot(predicted_load_values, label='Predicted Load Demand', color='red')
plt.title('Actual vs. Predicted Load Demand(BSES RAJDHANI)')
plt.xlabel('Time')
plt.ylabel('Load Demand')
plt.legend()
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame containing the "DATE/TIME" column
# Assuming 'predictions_original_scale' is a numpy array of predicted values
# Assuming 'y_test' is a numpy array of actual values
# Assuming 'split_index' is the index where you split your data into training and testing sets

# Extract the "DATE/TIME" column
dates = data.index[split_index + sequence_length:]  # Assuming the date is the index of your DataFrame

# # Reshape y_test to match the length of dates
# y_test_reshaped = y_test.reshape(-1)
# dates = df['DATE/TIME'][split_index + sequence_length:].values

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(dates, actual_values_original_scale, label='Actual')
plt.plot(dates, predicted_load_values, label='Predicted')
plt.xlabel('Date/Time')
plt.ylabel('Load Demand')
plt.title('Actual vs Predicted Load Demand(BSES RAJDHANI)')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()
