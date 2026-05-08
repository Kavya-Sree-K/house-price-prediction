# House Price Prediction Using Machine Learning

## Project Overview

This project focuses on predicting house prices using Machine Learning techniques based on different housing features such as living area, overall quality, garage area, basement size, neighborhood, house style, and many other numerical and categorical attributes. The complete workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, outlier removal, feature selection, encoding, scaling, and building multiple regression models to compare performance.

The main objective of this project is to understand how different house-related features affect the final sale price of a property and to build an accurate regression model capable of predicting house prices effectively.

This project was implemented using Python and various Data Science libraries including Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, and XGBoost.

---

# Problem Statement

House prices depend on multiple factors such as:

* Size of the house
* Living area
* Basement area
* Garage size
* Neighborhood
* House quality
* Year built
* Remodeling year
* House style
* Many other structural and location-based features

Manually estimating prices based on all these factors is difficult. Therefore, this project uses Machine Learning algorithms to analyze patterns from historical housing data and predict house prices automatically.

---

# Technologies and Libraries Used

## Programming Language

* Python

## Libraries Used

* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-learn
* XGBoost
* SciPy

---

# Dataset Description

The dataset contains both numerical and categorical features related to houses.

Some important features include:

| Feature      | Description                         |
| ------------ | ----------------------------------- |
| SalePrice    | Target variable (House Price)       |
| GrLivArea    | Above ground living area            |
| LotArea      | Lot size of property                |
| OverallQual  | Overall material and finish quality |
| TotalBsmtSF  | Total basement area                 |
| GarageArea   | Garage area                         |
| Neighborhood | Location of house                   |
| HouseStyle   | Style of house                      |
| YearRemodAdd | Year remodeled                      |
| YrSold       | Year sold                           |

The dataset contains both:

* Numerical columns
* Categorical columns

---

# Step-by-Step Workflow

# Step 1: Importing Required Libraries

Initially, important Python libraries were imported for:

* Data handling
* Visualization
* Model building
* Data preprocessing
* Evaluation

Libraries used:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

---

# Step 2: Loading the Dataset

The dataset was loaded using Pandas.

```python
df = pd.read_csv('housesing_price.csv')
```

After loading the dataset:

* Shape of dataset was checked
* Number of rows and columns were analyzed

---

# Step 3: Splitting Numerical and Categorical Columns

The dataset was divided into:

## Numerical Features

Features having integer or float values.

## Categorical Features

Features having object/string values.

This separation helps in applying different preprocessing techniques.

---

# Step 4: Exploratory Data Analysis (EDA)

Detailed exploratory data analysis was performed to understand the dataset.

## Numerical Analysis

Statistical information such as:

* Mean
* Median
* Standard deviation
* Minimum value
* Maximum value

was analyzed using:

```python
df.describe()
```

## Categorical Analysis

Frequency distribution of categorical columns was analyzed.

---

# Step 5: Checking Duplicate Values

Duplicate rows were checked using:

```python
df.duplicated().sum()
```

No duplicate values were found in the dataset.

---

# Step 6: Visualizing Target Variable

A histogram plot was used to visualize the distribution of the target variable `SalePrice`.

```python
sns.histplot(df['SalePrice'], kde=True)
```

Purpose:

* Understand data distribution
* Detect skewness
* Analyze price spread

---

# Step 7: Handling Missing Values

Missing values were analyzed separately for:

* Numerical columns
* Categorical columns

## Numerical Missing Values

Missing numerical values were replaced using the median.

Reason:

* Median is robust to outliers
* Better than mean for skewed data

```python
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
```

## Categorical Missing Values

Columns with more than 50% missing values were dropped.

Dropped columns:

* PoolQC
* MiscFeature
* Alley
* Fence
* MasVnrType
* FireplaceQu

Reason:

* Too many missing values reduce reliability
* Imputation may introduce noise

Remaining categorical missing values were filled using mode.

Reason:

* Mode represents most frequent category
* Suitable for categorical features

---

# Step 8: Extracting Insights from Data

Several business insights were extracted from the dataset.

## Insights Performed

* Average price based on overall quality
* Top 10 expensive houses
* Top 10 cheapest houses
* Houses with largest living area
* Houses with highest quality
* Average and median sale prices
* Sales trend by month
* Sales trend by year
* Remodel year effect on prices
* Average price by house style

Visualization techniques used:

* Bar plots
* Scatter plots
* Histograms
* Line charts
* Box plots

Purpose:

* Understand feature importance
* Analyze relationships
* Detect patterns in house prices

---

# Step 9: Outlier Detection and Removal

Outliers were detected using the IQR (Interquartile Range) method.

Important numerical columns considered:

* SalePrice
* GrLivArea
* LotArea
* TotalBsmtSF
* 1stFlrSF
* GarageArea

## Why Outlier Removal?

Outliers can:

* Reduce model accuracy
* Distort learning patterns
* Affect regression performance

After detecting outliers:

* Extreme values were removed
* Boxplots were used before and after removal

This improved data quality significantly.

---

# Step 10: Feature Engineering

A new feature named `TotalSF` was created.

```python
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
```

## Why Feature Engineering?

Feature engineering helps the model understand data better.

Instead of separate floor areas, a combined total square feet feature gives:

* Better representation
* Better relationship with house price
* Improved model learning

After creating the feature, unnecessary columns were removed.

---

# Step 11: Correlation Analysis

Correlation analysis was performed on numerical columns.

```python
df_num_corr = numerical_df.corr()
```

Features highly correlated with `SalePrice` were selected.

Threshold used:

```python
Correlation > 0.3 or < -0.3
```

## Why Correlation Analysis?

Purpose:

* Identify strong predictors
* Remove weak features
* Improve model efficiency
* Reduce noise

---

# Step 12: Feature Selection for Categorical Columns

ANOVA statistical testing was used for categorical feature selection.

```python
from scipy.stats import f_oneway
```

Features with:

```python
p-value < 0.05
```

were selected.

## Why ANOVA?

ANOVA helps determine whether categorical features significantly affect house prices.

Low p-value means:

* Feature has strong impact
* Feature is statistically significant

This helps improve model performance by selecting important categorical variables.

---

# Step 13: Feature Scaling

Numerical features were scaled using:

```python
StandardScaler
```

## Why Scaling?

Machine Learning algorithms perform better when features are on similar scales.

Scaling helps:

* Faster convergence
* Better optimization
* Improved model stability
* Better performance for distance-based models

---

# Step 14: One-Hot Encoding

Categorical features were converted into numerical format using:

```python
pd.get_dummies()
```

## Why Encoding?

Machine Learning models cannot directly understand text values.

Example:

| Neighborhood |
| ------------ |
| NAmes        |
| CollgCr      |
| OldTown      |

were converted into binary numerical columns.

This process allows models to use categorical information effectively.

---

# Step 15: Combining Final Features

Scaled numerical features and encoded categorical features were combined into a final dataset.

```python
X = pd.concat([X_num_scaled, cat_encoded], axis=1)
```

Target variable:

```python
y = df_clean['SalePrice']
```

---

# Step 16: Train-Test Split

Dataset was split into:

* Training data
* Testing data

Purpose:

* Train model on training data
* Evaluate model on unseen testing data

This helps measure real-world performance.

---

# Models Used in This Project

Multiple regression models were used and compared.

---

# 1. Linear Regression

## Why Linear Regression?

Linear Regression is a simple baseline regression algorithm.

It helps:

* Understand linear relationships
* Measure baseline performance
* Interpret feature influence

## Advantages

* Simple and fast
* Easy to interpret
* Good baseline model

## Limitations

* Cannot capture complex nonlinear relationships
* Sensitive to outliers

The model was trained using:

```python
from sklearn.linear_model import LinearRegression
```

Evaluation metrics used:

* R2 Score
* MSE
* RMSE

Scatter plots were used to compare:

* Actual prices
* Predicted prices

---

# 2. Random Forest Regressor

## Why Random Forest?

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

It was used because:

* Housing data contains nonlinear relationships
* Random Forest handles complex patterns effectively
* Robust against overfitting
* Handles large feature sets well

## Advantages

* High accuracy
* Handles nonlinear data
* Reduces overfitting
* Works well on tabular datasets

The model was trained using:

```python
from sklearn.ensemble import RandomForestRegressor
```

---

# 3. Support Vector Regressor (SVR)

## Why SVR?

SVR was used to capture nonlinear relationships in the dataset.

GridSearchCV was used for hyperparameter tuning.

## Advantages

* Effective for nonlinear regression
* Good generalization capability
* Works well with scaled features

## Why Parameter Tuning?

Hyperparameter tuning helps:

* Improve model accuracy
* Find optimal parameter combinations
* Reduce prediction errors

Pipeline and GridSearchCV were used.

---

# 4. XGBoost Regressor

## Why XGBoost?

XGBoost is one of the most powerful boosting algorithms for structured/tabular datasets.

It was selected because:

* High prediction accuracy
* Handles complex feature interactions
* Efficient and scalable
* Performs exceptionally well in regression problems

## Parameters Used

```python
n_estimators = 500
learning_rate = 0.05
max_depth = 4
subsample = 0.8
colsample_bytree = 0.8
```

## Advantages

* Very high performance
* Handles missing patterns well
* Reduces overfitting using boosting
* Excellent for competitions and real-world problems

XGBoost produced strong predictive performance for house price prediction.

---

# Evaluation Metrics Used

The following evaluation metrics were used:

| Metric   | Purpose                  |
| -------- | ------------------------ |
| R2 Score | Measures goodness of fit |
| MSE      | Mean Squared Error       |
| RMSE     | Root Mean Squared Error  |
| MAE      | Mean Absolute Error      |

## Why Multiple Metrics?

Different metrics provide different insights:

* R2 Score measures overall prediction quality
* MSE penalizes larger errors
* RMSE gives error in original units
* MAE measures average prediction error

---

# Data Visualization Performed

Several visualizations were created during the project:

* Histogram plots
* Scatter plots
* Box plots
* Bar charts
* Correlation analysis
* Actual vs Predicted plots

Purpose of visualization:

* Understand data distribution
* Detect outliers
* Identify feature relationships
* Evaluate model performance

---

# Key Learning Outcomes

Through this project, the following concepts were learned and implemented:

* Data preprocessing
* Handling missing values
* Feature engineering
* Exploratory Data Analysis
* Outlier removal
* Correlation analysis
* Statistical feature selection
* Feature scaling
* Encoding categorical variables
* Regression modeling
* Ensemble learning
* Hyperparameter tuning
* Model evaluation

---

# Conclusion

This project successfully implemented a complete Machine Learning pipeline for house price prediction.

Different regression algorithms were trained and evaluated to understand their performance on housing data. Extensive preprocessing, feature engineering, and feature selection techniques significantly improved model quality.

Among all models, advanced ensemble models such as Random Forest and XGBoost performed better because they can capture nonlinear relationships and complex interactions between features.

This project demonstrates the importance of:

* Proper data preprocessing
* Feature engineering
* Model comparison
* Statistical analysis
* Hyperparameter tuning

in building accurate Machine Learning solutions for real-world prediction problems.

---

# Future Improvements

Possible future enhancements include:

* Deep Learning approaches
* Advanced feature engineering
* Cross-validation optimization
* Deployment using Flask or Streamlit
* Real-time house price prediction web application
* Automated hyperparameter optimization

---

# Author

Kodumuri Kavyasree

Final Year BTech CSE Student

Interested in:

* Machine Learning
* Full Stack Development
* Data Science
* Cybersecurity
