import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ANN_data.csv', parse_dates=['DATE/TIME'])
X=df.drop(['DATE/TIME','BSES RAJDHANI','BSES YAMUNA','TPDDL','NDMC','MES'], axis=1)
y = df[['BSES RAJDHANI']]
y = pd.DataFrame(y)
graph_date = df[['DATE/TIME']][210240:210527]
print(X.head())
print(y.head())
print(graph_date)

date_column = df['DATE/TIME']
print(date_column)

# X contains both numeric and datetime columns
# Exclude non-numeric or datetime columns

numeric_columns = X.select_dtypes(include=[np.number]).columns
X_numeric = X[numeric_columns]

# Do the same for the target variable y
numeric_columns_y = y.select_dtypes(include=[np.number]).columns
y_numeric = y[numeric_columns_y]

#Scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_numeric = imputer.fit_transform(X_numeric)
y_numeric = imputer.fit_transform(y_numeric)

sc_X = MinMaxScaler(feature_range=(0,1))
X_scaled = sc_X.fit_transform(X_numeric)
X_scaled = pd.DataFrame(X_scaled)

sc_y = MinMaxScaler()
y_scaled = sc_y.fit_transform(y_numeric)
y_scaled = pd.DataFrame(y_scaled)
print(y_scaled)

graph_date = X_scaled[210240:210527]
graph_date.columns = X.columns[:]
print(graph_date)

print(X_scaled)
print(y_scaled)

graph_target = y_scaled[210240:210527]
print(graph_target)

#feature selection
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np

estimator = LinearRegression()
list_r2 = []
max_r2 = 0
selected_features = None

for i in range(1,len(X_scaled.columns)+1):
    selector = RFE(estimator=estimator, n_features_to_select=i, step=1)
    selector = selector.fit(X_numeric, y_scaled)
    adj_r2 = 1-((len(X_numeric)-1) / (len(X_numeric)-i-1)) * (1-selector.score(X_numeric, y_scaled))
    list_r2.append(adj_r2)

    if max_r2 < adj_r2:
        selected_features = selector.support_
        max_r2 = adj_r2

# 'selected_features' now contains the selected features as a boolean mask
# You can use it to subset your X_scaled DataFrame

X_selected = X_scaled[X_scaled.columns[selected_features]]
print(X_selected)

# Combine the selected features with the 'date_column' again
X_with_date = pd.concat([X_selected,date_column], axis = 1)
# Print the selected features
print(X_with_date.columns)
print(X_with_date)

X_original_column_names = X.columns[:]
# Extract the selected feature names from the original column names
selected_feature_names = [X_original_column_names[i] for i, is_selected in enumerate(selected_features) if is_selected]
selected_feature_names.append('DATE/TIME')
print(len(selected_feature_names))
# Rename the columns of X_with_date
X_with_date.columns = selected_feature_names

# Now, X_with_date will have the selected feature names as its column names
print(X_with_date.head())

y_original_column_names = y.columns
y_scaled.columns = y_original_column_names
print(y_scaled)

'''corelation'''
# X=X_with_date.drop('DATE/TIME',axis=1)
cor_matrix=X.astype(float).corr(method='pearson')


'''split data'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_with_date,y_scaled,test_size=1/3,random_state=0)

print(X_train)
X_train=X_train.drop(['DATE/TIME'], axis=1)
print(X_train)

test_date=X_test[['DATE/TIME']]
print(test_date)

X_test=X_test.drop(['DATE/TIME'], axis=1)
print(X_test)

print(y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
model = LinearRegression()
model.fit(X_train, y_train) # train the model
y_pred = model.predict(X_test) #make predictions on the testing set

#evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(y_pred)

y_pred = sc_y.inverse_transform(y_pred)
print(y_pred)

X_test_original = sc_X.inverse_transform(X_test)
X_input=pd.DataFrame(X_test_original, columns=X_original_column_names )

y_test_original=sc_y.inverse_transform(y_test)
y_input= pd.DataFrame(y_test_original, columns=y_original_column_names )

print(y_input)

y_predicted= pd.DataFrame(y_pred, columns=y_original_column_names )
print(y_predicted)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

import matplotlib.pyplot as plt

# Inverse transform the actual values to the original scale
# actual_values_original_scale = scaler.inverse_transform(y_test)

# Plotting the actual values vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Actual Load Demand', color='blue')
plt.plot(y_pred, label='Predicted Load Demand', color='red')
# plt.plot(actual_values_original_scale, label='Actual Values', linestyle='-', marker='o', color='b')
# plt.plot(predictions_original_scale, label='Predicted Values', linestyle='-', marker='X', color='r')
plt.title('Actual vs. Predicted Load Demand(BSES RAJDHANI)')
plt.xlabel('Time')
plt.ylabel('Load Demand')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# Inverse transform the actual values to the original scale
# actual_values_original_scale = scaler.inverse_transform(y_test)

# Plotting the actual values vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test_original[-200:], label='Actual Load Demand', color='blue')
plt.plot(y_pred[-200:], label='Predicted Load Demand', color='red')
# plt.plot(actual_values_original_scale, label='Actual Values', linestyle='-', marker='o', color='b')
# plt.plot(predictions_original_scale, label='Predicted Values', linestyle='-', marker='X', color='r')
plt.title('Actual vs. Predicted Load Demand(BSES RAJDHANI)')
plt.xlabel('Time')
plt.ylabel('Load Demand')
plt.legend()
plt.show()