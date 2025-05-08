import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Part 1 - Inspection and Cleaning
# Read the data
df = pd.read_csv('../data/housing.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# Create histograms for all numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    df[col].hist(bins=50)
    plt.title(col)
plt.tight_layout()
plt.show()

# Handle NaN values
print("\nNumber of NaN values in each column:")
print(df.isnull().sum())

# Fill NaN values in total_bedrooms with median
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# Create new features
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['population_per_household'] = df['population'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']

# Remove outliers in rooms_per_household
df = df[df['rooms_per_household'] < 20]

# Part 2 - Exploratory Data Analysis
# Distribution of median house value
plt.figure(figsize=(10, 6))
df['median_house_value'].hist(bins=100)
plt.title('Distribution of Median House Value')
plt.show()

# Correlation analysis
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Joint plot of median house value and median income
sns.jointplot(x='median_income', y='median_house_value', data=df, kind='reg')
plt.show()

sns.jointplot(x='median_income', y='median_house_value', data=df, kind='kde')
plt.show()

# Create income categories
df['income_cat'] = pd.qcut(df['median_income'], 
                          q=[0, 0.25, 0.5, 0.75, 0.95, 1],
                          labels=['Low', 'Below_Average', 'Above_Average', 'High', 'Very High'])

# Plot count of income categories by ocean proximity
plt.figure(figsize=(12, 6))
sns.countplot(x='income_cat', hue='ocean_proximity', data=df)
plt.title('Income Categories by Ocean Proximity')
plt.show()

# Bar plots for median house value
plt.figure(figsize=(12, 6))
sns.barplot(x='income_cat', y='median_house_value', data=df)
plt.title('Median House Value by Income Category')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='ocean_proximity', y='median_house_value', data=df)
plt.title('Median House Value by Ocean Proximity')
plt.show()

# Create pivot table and heatmap
pivot_table = pd.pivot_table(df, 
                            values='median_house_value',
                            index='income_cat',
                            columns='ocean_proximity',
                            aggfunc='count')
pivot_table = pivot_table.drop('ISLAND', axis=1)

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Count of Houses by Income Category and Ocean Proximity')
plt.show()

# Part 3 - Preparing Data
# Drop income_cat column
df = df.drop('income_cat', axis=1)

# Standardize numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
z_scored = df[numeric_cols].apply(lambda x: stats.zscore(x))

# Create dummy variables for categorical columns
dummies = pd.get_dummies(df['ocean_proximity'], drop_first=True)

# Prepare target variable
y = df['median_house_value']

# Prepare features
X = pd.concat([z_scored.drop('median_house_value', axis=1), dummies], axis=1)

# Part 4 - Machine Learning
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on training data
train_predictions = lr_model.predict(X_train)

# Plot training predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_predictions)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Training Predictions vs Actual Values')
plt.show()

# Calculate metrics for training data
train_mse = metrics.mean_squared_error(y_train, train_predictions)
train_r2 = metrics.r2_score(y_train, train_predictions)
print(f"\nTraining Metrics:")
print(f"Mean Squared Error: {train_mse}")
print(f"R2 Score: {train_r2}")

# Make predictions on test data
test_predictions = lr_model.predict(X_test)

# Plot test predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Test Predictions vs Actual Values')
plt.show()

# Calculate metrics for test data
test_mse = metrics.mean_squared_error(y_test, test_predictions)
test_r2 = metrics.r2_score(y_test, test_predictions)
print(f"\nTest Metrics:")
print(f"Mean Squared Error: {test_mse}")
print(f"R2 Score: {test_r2}")

# Calculate RMSE
rmse = np.sqrt(test_mse)
print(f"Root Mean Squared Error: {rmse}")

# Bonus Question 1
# Create dataframe with actual and predicted values
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': test_predictions
})

# Calculate absolute errors
absolute_errors = abs(results_df['Actual'] - results_df['Predicted'])
mean_absolute_error = absolute_errors.mean()
print(f"\nMean Absolute Error: {mean_absolute_error}")

# Bonus Question 2 - Random Forest
# Create and train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
rf_train_predictions = rf_model.predict(X_train)
rf_test_predictions = rf_model.predict(X_test)

# Calculate metrics for Random Forest
rf_train_mse = metrics.mean_squared_error(y_train, rf_train_predictions)
rf_train_r2 = metrics.r2_score(y_train, rf_train_predictions)
rf_test_mse = metrics.mean_squared_error(y_test, rf_test_predictions)
rf_test_r2 = metrics.r2_score(y_test, rf_test_predictions)

print("\nRandom Forest Metrics:")
print(f"Training MSE: {rf_train_mse}")
print(f"Training R2: {rf_train_r2}")
print(f"Test MSE: {rf_test_mse}")
print(f"Test R2: {rf_test_r2}") 