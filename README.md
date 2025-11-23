# Car_Price_Prediction

**Car Price Prediction using Machine Learning**

This project builds a Machine Learning model to predict the selling price of used cars based on various features such as fuel type, transmission, seller type, kilometers driven, and car age.
It includes data cleaning, preprocessing, visualization, model training, evaluation, and feature importance analysis.

**Project Overview**

The goal of this project is to develop an accurate model that can estimate the price of a used car using regression algorithms.
The steps include:

* Importing and preparing the dataset
* Data exploration and cleaning
* Visualizing key relationships
* Feature engineering
* Converting categorical variables
* Splitting data
* Training multiple ML models
* Choosing the best model
* Predicting & evaluating
* Identifying important features

**Technologies & Libraries Used**

_**Python Libraries**_

* NumPy – numerical operations
* Pandas – data loading & preprocessing
* Matplotlib / Seaborn – data visualization
* Scikit-Learn
    - train_test_split
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    - Model evaluation metrics (r2_score, mean_squared_error)
 
**Dataset Information**

The dataset contains details such as:

* Year of manufacture
* Present_Price (current ex-showroom price)
* Selling_Price (target variable)
* Kms_Driven
* Fuel_Type (Petrol/Diesel/CNG)
* Seller_Type (Dealer/Individual)
* Transmission
* Owner

**Data Preprocessing**

1. Loaded dataset and displayed initial samples
2. Checked missing values, data types, duplicates
3. Removed duplicate rows
4. Dropped unnecessary columns (e.g., Car_Name)
5. Cleaned column names
6. Created new feature ➝ Car_Age = max(Year) - Year
7. Dropped the original Year column
8. Converted categorical variables to numerical using get_dummies()

**Data Visualization**

The project visualizes relationships between features and target variable:

_**Categorical vs Selling Price**_

* Fuel Type vs Selling Price
* Transmission vs Selling Price
* Seller Type vs Selling Price

_**Numerical Correlations**_

* Present Price vs Selling Price
* Kms Driven vs Selling Price

_**Actual vs Predicted graph after final model**_

A scatterplot to compare real vs predicted values.

_**Feature Importance**_

Top 10 features influencing the price (from Random Forest model).

**Machine Learning Models Used**

Three regression models were trained and compared:
* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

_Each model was evaluated using:_

* R² Score
* RMSE (Root Mean Squared Error)

_**Best Performing Model**_

The Random Forest Regressor achieved the highest accuracy and lowest RMSE, so it was selected as the final model.

**Model Evaluation**

Using the best model (Random Forest):

* Plotted Actual vs Predicted Selling Prices
* Printed final R² score and RMSE
* Generated feature importance ranking
