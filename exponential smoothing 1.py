import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ANN_data.csv', parse_dates=['DATE/TIME'] )


X=df[['DATE/TIME','BSES RAJDHANI']]
print(X.head())

date_column=df['DATE/TIME']
print(date_column)

numeric_columns = X.select_dtypes(include=[np.number]).columns
X_numeric = X[numeric_columns]
print(X_numeric)

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_numeric = imputer.fit_transform(X_numeric)
# y_numeric = imputer.fit_transform(y_numeric)

sc_X = MinMaxScaler()
X_scaled = sc_X.fit_transform(X_numeric)
X_scaled=pd.DataFrame(X_scaled)
print(X_scaled)

X_scaled.columns = ['BSES RAJDHANI']
# Combine the selected features with the 'date_column' again
X_with_date = pd.concat([X_scaled, date_column], axis=1)

print(X_with_date)
X_train=X_with_date[:-200]
print(X_train)
X_test=X_with_date[-200:]

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

predictions = pd.DataFrame()
model = ExponentialSmoothing(X_train['BSES RAJDHANI'], seasonal='add', seasonal_periods = 24*60)
fit_model = model.fit()

# predict on the test set
predictions['BSES RAJDHANI'] = fit_model.forecast(len(X_test))

mse = mean_squared_error(X_test['BSES RAJDHANI'], predictions)
print(f'Mean Squared Error for BSES RAJADHANI: {mse}')

predicted_load_values=sc_X.inverse_transform(predictions)
print(X_test)

datetime_columns = ['DATE/TIME']
X_test_numeric = X_test.drop(datetime_columns, axis=1)

actual_values_original_scale = sc_X.inverse_transform(X_test_numeric)
print(actual_values_original_scale)
print(predicted_load_values)

import matplotlib.pyplot as plt

# Inverse transform the actual values to the original scale
# actual_values_original_scale = scaler.inverse_transform(y_test)

# Plotting the actual values vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(actual_values_original_scale[:200], label='Actual Load Demand', color='blue')
plt.plot(predicted_load_values[:200], label='Predicted Load Demand', color='red')
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