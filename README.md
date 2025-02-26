# bankruptcy_prediction
# Bankruptcy Prevention using Machine Learning

## Overview
This project focuses on predicting bankruptcy risk using machine learning techniques. The goal is to build a classification model that helps identify financially unstable firms based on various financial indicators.

Bankruptcy prediction is crucial for investors, creditors, and financial analysts to assess the financial health of companies and take necessary measures to mitigate risks.

## Dataset
The dataset contains various financial indicators of companies, including:
- industrial_risk
- management_risk
- financial_flexibility
- credibility
- competitiveness
-  operating_risk
-  
These features help in predicting whether a company is at risk of bankruptcy.

## Project Workflow

### 1. Install Dependencies
Ensure the necessary libraries are installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Load and Explore Data
The dataset is loaded and examined for missing values, outliers, and feature distributions.
```python
import pandas as pd
import seaborn as sns

# Load dataset
data = pd.read_csv("bankruptcy_data.csv")

# Display basic information
data.info()
data.describe()
```

### 3. Data Preprocessing
Preprocessing steps include:
- Handling missing values
- Encoding categorical variables
- Normalizing numerical features
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['Bankrupt']))
```

### 4. Model Training
Various classification models are trained, including:
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Neural Networks
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_scaled, data['Bankrupt'], test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 5. Model Evaluation
Performance is assessed using accuracy, precision, recall, and F1-score.
```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Results
The best-performing model achieves high accuracy and recall, making it a valuable tool for bankruptcy prediction.

## How to Use
The trained model can be used to predict bankruptcy risk for new financial data:
```python
new_data = [[0.5, -1.2, 1.8, 0.3, 2.0]]  # Example financial ratios
prediction = model.predict(new_data)
print("Bankruptcy Risk:" , "Yes" if prediction[0] == 1 else "No")
```


## GitHub Repository
The project is available on GitHub:
[Bankruptcy-Prevention](https://github.com/sourabhmatali/Bankruptcy-Prevention)

