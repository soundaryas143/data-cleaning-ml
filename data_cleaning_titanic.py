# ===============================
# Task 1: Data Cleaning & Preprocessing
# Dataset: Titanic (from seaborn)
# Tools: Python, Pandas, NumPy, Matplotlib, Seaborn
# ===============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 2. Load Dataset
titanic = sns.load_dataset('titanic')

print("----- First 5 Rows -----")
print(titanic.head())
print("\n----- Dataset Info -----")
print(titanic.info())
print("\n----- Missing Values -----")
print(titanic.isnull().sum())

# 3. Handle Missing Values
# Fill 'age' missing values with median
titanic['age'] = titanic['age'].fillna(titanic['age'].median())

# Fill 'embarked' with most frequent value
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])

# Drop 'deck' column (too many missing values)
titanic = titanic.drop(columns=['deck'])

# 4. Convert Categorical → Numerical
titanic = pd.get_dummies(titanic, columns=['sex', 'embarked', 'class'], drop_first=True)

# Convert boolean columns to int
titanic['alone'] = titanic['alone'].astype(int)
titanic['adult_male'] = titanic['adult_male'].astype(int)
titanic['who'] = titanic['who'].map({'man': 0, 'woman': 1, 'child': 2})

# 5. Normalize / Standardize Numerical Features
scaler = StandardScaler()
titanic[['age', 'fare']] = scaler.fit_transform(titanic[['age', 'fare']])

# 6. Detect & Remove Outliers (Example on 'fare')
plt.figure(figsize=(6,4))
sns.boxplot(x=titanic['fare'])
plt.title("Boxplot - Fare (Before Outlier Removal)")
plt.show()

# Outlier removal using IQR
Q1 = titanic['fare'].quantile(0.25)
Q3 = titanic['fare'].quantile(0.75)
IQR = Q3 - Q1
titanic = titanic[(titanic['fare'] >= Q1 - 1.5*IQR) & (titanic['fare'] <= Q3 + 1.5*IQR)]

plt.figure(figsize=(6,4))
sns.boxplot(x=titanic['fare'])
plt.title("Boxplot - Fare (After Outlier Removal)")
plt.show()

# 7. Final Check
print("\n----- Cleaned Dataset Info -----")
print(titanic.info())
print("\n----- First 5 Rows (Cleaned) -----")
print(titanic.head())
