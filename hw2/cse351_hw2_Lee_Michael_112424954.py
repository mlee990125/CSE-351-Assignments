#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[2]:


# Read the datasets
weather_data = pd.read_csv('weather_data.csv')
energy_usage_data = pd.read_csv('energy_data.csv')


# In[3]:


# weather_data
(weather_data)


# In[4]:


# energy_usage_data
(energy_usage_data)


# In[5]:


# Q1.
# Parse time fields
weather_data['time'] = pd.to_datetime(weather_data['time'], unit='s')
energy_usage_data['Date & Time'] = pd.to_datetime(energy_usage_data['Date & Time'])

# Calculate daily energy usage
energy_usage_data['date'] = energy_usage_data['Date & Time'].dt.date
daily_energy_usage = energy_usage_data.groupby('date')['use [kW]'].sum().reset_index()

# Convert datetime column to same data type
daily_energy_usage['date'] = pd.to_datetime(daily_energy_usage['date'])

# Merge datasets
merged_data = pd.merge(weather_data, daily_energy_usage, left_on='time', right_on='date')


# In[6]:


(merged_data)


# In[7]:


# Q2. 
# Filter the merged_data to only include data before the month of December
train_data = merged_data.loc[(merged_data['date'].dt.month != 12)]

# Create a test set with only December data
test_data = merged_data.loc[(merged_data['date'].dt.month == 12)]


# In[8]:


train_data


# In[9]:


test_data


# In[10]:


# Separate features (X) and target (y) for both training and testing sets
X_train = train_data.drop(['date', 'use [kW]', 'time'], axis=1)
y_train = train_data['use [kW]']
X_test = test_data.drop(['date', 'use [kW]', 'time'], axis=1)
y_test = test_data['use [kW]']


# In[11]:


print(X_train.dtypes)
print(X_test.dtypes)


# In[12]:


# One-hot encode the categorical columns in both train and test sets
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Ensure both train and test sets have the same columns after one-hot encoding
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, axis=1, fill_value=0)


# In[13]:


# Q3. 
# Instantiate the LinearRegression model
lr_model = LinearRegression()


# In[14]:


# Fit the model to the training data
lr_model.fit(X_train_encoded, y_train)


# In[15]:


# Make predictions on the test set
y_pred = lr_model.predict(X_test_encoded)

# Calculate the root mean squared error
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse}")


# In[16]:


# Create a scatter plot of the actual vs. predicted energy usage values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Energy Usage (kW)')
plt.ylabel('Predicted Energy Usage (kW)')
plt.title('Actual vs. Predicted Energy Usage (kW)')

# Add a reference line representing a perfect prediction
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)

# Display the plot
plt.show()


# In[17]:


# Create a new DataFrame with the date and predicted values
pred_df = pd.DataFrame({'date': test_data['date'], 'predicted_value': y_pred})

# Save the DataFrame as a CSV file
pred_df.to_csv('cse351_hw2_Lee_Michael_112424954_linear_regression.csv', index=False)


# In[18]:


# Q4. 
# Create copies of train_data and test_data to avoid SettingWithCopyWarning
train_data = train_data.copy()
test_data = test_data.copy()

# Create a binary column 'is_high' in the train_data and test_data DataFrames
# Assign 1 if the temperature is greater than or equal to 35, and 0 otherwise
train_data['is_high'] = np.where(train_data['temperature'] >= 35, 1, 0)
test_data['is_high'] = np.where(test_data['temperature'] >= 35, 1, 0)


# In[19]:


# Prepare features (X) and target (y) for both training and testing sets
X_train = train_data.drop(['date', 'is_high', 'temperature'], axis=1)
y_train = train_data['is_high']
X_test = test_data.drop(['date', 'is_high', 'temperature'], axis=1)
y_test = test_data['is_high']


# In[20]:


# Preprocess your features (e.g., One-Hot Encoding for categorical variables)
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)


# In[21]:


# Instantiate a LogisticRegression model
logistic_model = LogisticRegression()


# In[22]:


# Fit the model to the training data
logistic_model.fit(X_train_encoded, y_train)


# In[23]:


# Make predictions on the test data
y_pred = logistic_model.predict(X_test_encoded)


# In[24]:


# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[25]:


# Caclculate F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)


# In[26]:


# Create a new DataFrame with date and predicted classification
classification_output = pd.DataFrame({'date': test_data['date'], 'classification': y_pred})

# Save the DataFrame to a CSV file
classification_output.to_csv('cse351_hw2_Lee_Michael_112424954_logistic_regression.csv', index=False)


# In[27]:


# Q5. 
# Filter energy usage data for day and night periods
day_data = energy_usage_data.loc[(energy_usage_data['Date & Time'].dt.hour >= 6) & (energy_usage_data['Date & Time'].dt.hour < 19)]
night_data = energy_usage_data.loc[(energy_usage_data['Date & Time'].dt.hour < 6) | (energy_usage_data['Date & Time'].dt.hour >= 19)]


# In[28]:


# Calculate average usage for each device during day and night periods
avg_washer_day = day_data['Washer [kW]'].mean()
avg_washer_night = night_data['Washer [kW]'].mean()


# In[29]:


avg_ac_day = day_data['AC [kW]'].mean()
avg_ac_night = night_data['AC [kW]'].mean()


# In[30]:


# Plot the results
fig, ax = plt.subplots()
devices = ['Washer', 'AC']
day_usage = [avg_washer_day, avg_ac_day]
night_usage = [avg_washer_night, avg_ac_night]

x = np.arange(len(devices))
width = 0.35

ax.bar(x - width/2, day_usage, width, label='Day')
ax.bar(x + width/2, night_usage, width, label='Night')

ax.set_ylabel('Average Energy Usage (kW)')
ax.set_title('Average Energy Usage of Washer and AC during Day and Night')
ax.set_xticks(x)
ax.set_xticklabels(devices)
ax.legend()

plt.show()


# In[31]:


# Analysis:
# It seems plausible that the washer is used more during the day than at night, but with a negligible difference
# because washer can be used during any time without preference. It seems it is used more during the day 
# because that is when people are most active.
# It is however, unexpected that AC energy usage is higher during the night time than day time. I have two ideas why
# this may be. 
# 1: It's possible that the location where the data was collected experiences hotter nights or high humidity levels 
# during the night, which could lead to increased AC usage to maintain comfortable conditions.
# 2: People may have preferences for cooler sleeping environments, which could lead to higher AC usage during the 
# night. 


# In[ ]:




