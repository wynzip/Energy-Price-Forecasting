import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss, adfuller
from scipy import stats
import warnings

# Authors: Sara Cupini, Francesco Panichi, Davide Pagani
# License: Apache-2.0 license
# Notice: this script is completely developed by us

warnings.filterwarnings("ignore")

def set_datetime_index(df, column_name):
    ts_price = df[column_name]
    ts_date = pd.to_datetime(df['Date'])
    for i in range(len(ts_date)):
        hours = i % 24
        ts_date[i] = ts_date[i] + pd.Timedelta(hours=hours)
    # ts_price_datetime = pd.Series(ts_price.tolist(), index=ts_date)
    return ts_date, ts_price

# Specify the path to your CSV file
file_path = "data/datasets/TrainingAndProtoTest.csv"

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

#Print the number of observations in the dataset
num_observations = len(df)
print('-' * 70)
print(f"Number of observations in the dataset: {num_observations}")
print('-' * 70)

#Print the names of the columns
column_names = df.columns.tolist()
print('-' * 70)
print("Names of the columns in the dataset:")
print(column_names)
print('-' * 70)

#Check for the presence of NaN values in each column
nan_details = df.isnull().sum()
print('-' * 70)
print("Details of NaN values for each column:")
print(nan_details)
print('-' * 70)
#  Slice the DataFrame to have the train set
date_column = df.columns[-1]
# Find the index of the row where the date is '2018-01-01'
cutoff_index = df[df[date_column] == '2018-01-01'].index[0]
df = df.iloc[:cutoff_index]

# Calculate the correlation matrix
correlation_matrix = df.iloc[:,:-1].corr()

# Display the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Calculate the mean for each column
mean_values = df.iloc[:,3:13].mean()
print('-' * 70)
print("Mean values for each column:")
for col, mean in mean_values.items():
    print(f"{col}: {mean}")
# Calculate the variance for each column
std_values = df.iloc[:,3:13].std()
print("\nStandard deviation values for each column:")
for col, std in std_values.items():
    print(f"{col}: {std}")
print('-' * 70)

# Build a copy of the DataFrame
df_ts=df.copy()
# Slice the DataFrame to have the train set
date_column_ts = df_ts.columns[-1]
# Find the index of the row where the date is '2016-01-01'
cutoff_index_ts = df_ts[df_ts[date_column_ts] == '2016-01-01'].index[0]

df_ts = df_ts.iloc[:cutoff_index_ts]



# Create the plot
plt.figure(figsize=(24, 12))

# Plot the EM Price
ts_date, ts_price = set_datetime_index(df_ts, 'TARG__EM_price')
plt.subplot(221)
plt.plot(ts_date, ts_price, label='Original Data', color='blue')
plt.title('Prices Time Series Data')
plt.xlabel('Date')
plt.ylabel('EM Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Plot the EM Load
ts_date, ts_price = set_datetime_index(df_ts, 'FUTU__EM_load_f')
plt.subplot(222)
plt.plot(ts_date, ts_price, label='Original Data', color='red')
plt.title('Load Time Series Data')
plt.xlabel('Date')
plt.ylabel('Load')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Plot the EM Wind
ts_date, ts_price = set_datetime_index(df_ts, 'FUTU__EM_wind_f')
plt.subplot(223)
plt.plot(ts_date, ts_price, label='Original Data', color='green')
plt.title('Wind Time Series Data')
plt.xlabel('Date')
plt.ylabel('Wind')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Plot the EM Solar
ts_date, ts_price = set_datetime_index(df_ts, 'FUTU__EM_solar_f')
plt.subplot(224)
plt.plot(ts_date, ts_price, label='Original Data', color='orange')
plt.title('Solar Time Series Data')
plt.xlabel('Date')
plt.ylabel('Solar')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Shapiro-Wilk test for Normal distribution
shapiro_stat, shapiro_p_value = stats.shapiro(df['TARG__EM_price'])
print('-' * 70)
print(f'Shapiro-Wilk Test: Statistic={shapiro_stat}, p-value={shapiro_p_value}')
if shapiro_p_value > 0.05:
    print('Normal distribution')
else:
    print('Not normal distribution')
print('-' * 70)

plt.figure(figsize=(12, 8))
sns.boxplot(y=df['TARG__EM_price'])
plt.title('TARG__EM_price boxplot')
plt.xlabel('')
plt.ylabel('')
plt.show()

#Plot the distribution of the 'TARG__EM_price' column
plt.figure(figsize=(10, 6))
sns.histplot(df['TARG__EM_price'], kde=True)
plt.title('Distribution of the TARG__EM_price column')
plt.xlabel('TARG__EM_price')
plt.ylabel('Frequency')
plt.show()

# Find the number of outliers
Q1 = df['TARG__EM_price'].quantile(0.25)
Q3 = df['TARG__EM_price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['TARG__EM_price'] < lower_bound) | (df['TARG__EM_price'] > upper_bound)]
num_outliers = len(outliers)
print('-' * 70)
print(f"Number of outliers in the Target column : {num_outliers}")
print('-' * 70)

# Study the trend and seasonality
ts_date, ts_price = set_datetime_index(df, 'TARG__EM_price')

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(ts_date, ts_price, label='Original Data', color='blue')
plt.title('Prices Time Series Data')
plt.xlabel('Date')
plt.ylabel('EM Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Perform seasonal decomposition
decomposition = seasonal_decompose(ts_price, model='additive', period=168)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the decomposed components
plt.figure(figsize=(20, 10))

plt.subplot(411)
plt.plot(ts_price, label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# Testing for Stationarity using statistical tests
# Dickey-Fuller Test
print('-' * 70)
print('Statistical tests - Complete Dataset' )
df_test = adfuller(ts_price, autolag='AIC')
print('Dickey-Fuller p-value: ' + str(df_test[1]))
# Kwiatkowski–Phillips–Schmidt–Shin (KPSS)
kpss_test = kpss(ts_price, nlags='auto')
print('KPSS p-value: ' + str(kpss_test[1]))
print('-' * 70)

# Calculate the hourly mean and standard deviation of the columns
# Initialize two lists to store mean and standard deviation DataFrames
means_list = []
stds_list = []

# Loop to load and analyze the 24 datasets
for hour in range(24):

    file_path = f"data/datasets/TrainingAndProtoTest_{hour}.csv"

    # Load the CSV file into a pandas DataFrame
    df_h = pd.read_csv(file_path)

    #  Slice the DataFrame to have the train set
    date_column_h = df_h.columns[-1]
    # Find the index of the row where the date is '2018-01-01'
    cutoff_index_h = df_h[df_h[date_column_h] == '2018-01-01'].index[0]

    df_h = df_h.iloc[:cutoff_index_h]

    # Calculate the mean and standard deviation for each column excluding the last one
    mean_values = df_h.iloc[:, 3].mean()
    std_values = df_h.iloc[:, 3].std()

    print('-' * 70)
    print(f'Mean and variance of Targ__EM_price - Dataset hour {hour}')
    print('Mean: ' + str(mean_values))
    print('Standard deviation: ' + str(std_values))

    # Append the mean and standard deviation DataFrames to the lists
    means_list.append(mean_values)
    stds_list.append(std_values)

    # Perform seasonal decomposition for each hour
    # Select column EM_price
    ts_price_h = df_h['TARG__EM_price']
    # Select dates
    ts_date_h = pd.to_datetime(df_h['Date'])


    # Set as index the datetime index
    ts_price_datetime_h = pd.Series(ts_price_h.tolist(), index=ts_date_h)

    # Testing for Stationarity using statistical tests
    print('-' * 70)
    print(f'Statistical tests - Dataset hour {hour}')
    # Dickey-Fuller Test
    df_test_h = adfuller(ts_price_h, autolag='AIC')
    print('Dickey-Fuller p-value: ' + str(df_test_h[1]))

    # Kwiatkowski–Phillips–Schmidt–Shin (KPSS)
    kpss_test_h = kpss(ts_price_h, nlags='auto')
    print('KPSS p-value: ' + str(kpss_test_h[1]))
    print('-' * 70)

# Define the hour
hour = list(range(24))

# Plot histogram of means and standard deviations
plt.figure(figsize=(10, 6))

# Plot means
plt.bar(hour, means_list, label='Mean')

# Plot standard deviations as error bars
plt.errorbar(hour, means_list, yerr=stds_list, fmt='o', color='red', capsize=5, label='Standard Deviation')

plt.title('Mean and Standard Deviation of TARG__EM_price for Each hour')
plt.xlabel('Hour')
plt.ylabel('Value')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()

# Evaluate mean and variance of each day of the week for the target
# Convert the date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract the day of the week (Monday=0, Sunday=6)
df['Day_of_week'] = df['Date'].dt.dayofweek

# Group by day of the week and calculate mean and standard deviation of column 4
means_day = df.groupby('Day_of_week')['TARG__EM_price'].mean()
stds_day = df.groupby('Day_of_week')['TARG__EM_price'].std()

# Dictionary to map day numbers to day names in English
days_of_week = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}

# Print the results
print('-' * 70)
print("Mean and Standard Deviation of TARG__EM_price for each day of the week:")

for day_num, mean_value in means_day.items():
    std_value = stds_day[day_num]
    day_name = days_of_week.get(day_num, f"Unknown day ({day_num})")
    print(f"{day_name}: Mean = {mean_value}, Std Dev = {std_value}")
print('-' * 70)
# Plot histograms for each day of the week

# Define the days of the week
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Plot histogram of means and standard deviations
plt.figure(figsize=(10, 6))

# Plot means
plt.bar(days_of_week, means_day, label='Mean')

# Plot standard deviations as error bars
plt.errorbar(days_of_week, means_day, yerr=stds_day, fmt='o', color='red', capsize=5, label='Standard Deviation')

plt.title('Mean and Standard Deviation of TARG__EM_price for Each Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Value')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()


# Evaluate mean and variance of week day and weekend day for the target
# Create a new column indicating whether it's a weekday or weekend
df['Weekday'] = df['Day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

# Group by weekday and weekend and calculate mean and standard deviation of 'TARGET_EM_price'
means_week = df.groupby('Weekday')['TARG__EM_price'].mean()
stds_week = df.groupby('Weekday')['TARG__EM_price'].std()
print('-' * 70)
print("\nMean and Standard Deviation of TARG__EM_price for Weekdays and Weekends:")

for period, mean_value in means_week.items():
    std_value = stds_week[period]
    print(f"{period}: Mean = {mean_value}, Std Dev = {std_value}")
print('-' * 70)
# Plot histograms for weekday and weekend
plt.figure(figsize=(10, 6))

# Plot weekday data
plt.bar('Weekday', means_week['Weekday'], yerr=stds_week['Weekday'], color='blue', alpha=0.7, label='Weekday')

# Plot weekend data
plt.bar('Weekend', means_week['Weekend'], yerr=stds_week['Weekend'], color='red', alpha=0.7, label='Weekend')

plt.title('Mean and Standard Deviation of TARG__EM_price for Weekdays and Weekends')
plt.ylabel('Mean')
plt.errorbar(['Weekday', 'Weekend'], means_week, yerr=stds_week, fmt='o', color='black', capsize=5, linestyle='None')
plt.legend()
plt.show()

