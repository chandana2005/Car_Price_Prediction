#importing all necessary libraries and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

#loading dataset and previewing
data = pd.read_csv('/content/car_price_presiction.csv')
print("Shape of data:", data.shape)
data.head()

#understanding the data
data.info()
print("\nMissing values:\n", data.isnull().sum())
print("\nDuplicate rows:", data.duplicated().sum())

#cleaing the data
data.drop_duplicates(inplace=True)
if 'Car_Name' in data.columns:
    data.drop('Car_Name', axis=1, inplace=True)
data.columns = data.columns.str.strip()
data.head()

#visualizing to understand what factors affect price
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.barplot(x='Fuel_Type', y='Selling_Price', data=data)
plt.title('Selling Price vs Fuel Type')

plt.subplot(1,3,2)
sns.barplot(x='Transmission', y='Selling_Price', data=data)
plt.title('Selling Price vs Transmission')

plt.subplot(1,3,3)
sns.barplot(x='Seller_Type', y='Selling_Price', data=data)
plt.title('Seller_Type vs Selling Type')

plt.tight_layout()
plt.show()

#visualizing to understand relationship between numerical data
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.regplot(x='Present_Price', y='Selling_Price', data=data)

plt.subplot(1,2,2)
sns.regplot(x='Kms_Driven', y='Selling_Price', data=data)

plt.tight_layout()
plt.show()


#determining car's age 
data['Car_Age'] = data['Year'].max() - data['Year']
data.drop('Year', axis=1, inplace=True)

data.head()

#converting categorical data into dummy values for ML model
data = pd.get_dummies(data, drop_first=True)
data.head()


#split dataset into testing and training
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

#train models and compare
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name} -> R2 Score: {r2:.3f}, RMSE: {rmse:.3f}")

#evaluate best model
best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price")
plt.show()

print("Final R2 Score:", r2_score(y_test, y_pred))
print("Final RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


#check which feature infulence price most
feat_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
feat_importance.head(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.show()

feat_importance.head(10)
