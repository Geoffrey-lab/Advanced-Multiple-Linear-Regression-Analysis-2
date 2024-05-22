# Advanced Multiple Linear Regression Analysis

This repository contains a Jupyter Notebook that provides an in-depth look at advanced multiple linear regression analysis using the `mtcars` dataset. The notebook covers various stages of regression analysis, including data exploration, model fitting, and diagnostic checks for model assumptions such as linearity, independence, homoscedasticity, and normality.

## Notebook Content Overview

### 1. Data Preparation and Exploration
- Loading the `mtcars` dataset.
- Displaying initial data and descriptive statistics.

### 2. Fitting the Multiple Linear Regression Model
- Generating the regression formula.
- Fitting the model using Ordinary Least Squares (OLS) with `statsmodels`.
- Displaying and interpreting the model summary.

### 3. Model Diagnostics
- Checking for independence: Residuals vs. predictor variable plots.
- Checking for homoscedasticity: Fitted values vs. residuals plot and Breusch-Pagan test.
- Checking for normality: Histogram and Q-Q plot of residuals.
- Identifying outliers: Cook's distance plot.

## Key Code Snippets

### Data Loading and Preparation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl

# Set the path to the CA certificates bundle
ssl._create_default_https_context = ssl._create_unverified_context

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/regression_sprint/mtcars.csv', index_col=0)
df.head(10)
```

### Data Exploration
```python
df.describe().T
```

### Fitting the Model Using `statsmodels.OLS`
```python
import statsmodels.formula.api as sm

# Generate the regression formula
formula_str = 'mpg ~ ' + ' + '.join(df.columns[1:])
print(formula_str)

# Construct and fit the model
model = sm.ols(formula=formula_str, data=df)
fitted = model.fit()

# Print the model summary
print(fitted.summary())
```

### Checking for Independence
```python
fig, axs = plt.subplots(2, 5, figsize=(14, 6), sharey=True)
fig.subplots_adjust(hspace=0.5, wspace=0.2)
fig.suptitle('Predictor variables vs. model residuals', fontsize=16)
axs = axs.ravel()

for index, column in enumerate(df.columns):
    axs[index-1].set_title("{}".format(column), fontsize=12)
    axs[index-1].scatter(x=df[column], y=fitted.resid, color='blue', edgecolor='k')
    axs[index-1].grid(True)
    xmin = min(df[column])
    xmax = max(df[column])
    axs[index-1].hlines(y=0, xmin=xmin*0.9, xmax=xmax*1.1, color='red', linestyle='--', lw=3)
    if index == 1 or index == 6:
        axs[index-1].set_ylabel('Residuals')
```

### Checking for Homoscedasticity
```python
plt.figure(figsize=(8, 3))
p = plt.scatter(x=fitted.fittedvalues, y=fitted.resid, edgecolor='k')
xmin = min(fitted.fittedvalues)
xmax = max(fitted.fittedvalues)
plt.hlines(y=0, xmin=xmin*0.9, xmax=xmax*1.1, color='red', linestyle='--', lw=3)
plt.xlabel("Fitted values", fontsize=15)
plt.ylabel("Residuals", fontsize=15)
plt.title("Fitted vs. residuals plot", fontsize=18)
plt.grid(True)
plt.show()
```

```python
import statsmodels.stats.api as sms

# Calculate residuals
residuals = fitted.resid

# Perform Breusch-Pagan test
bp_test_result = sms.het_breuschpagan(residuals, fitted.model.exog)
print("Breusch-Pagan Test Results:")
print("LM Statistic:", bp_test_result[0])
print("LM-Test p-value:", bp_test_result[1])
print("F-Statistic:", bp_test_result[2])
print("F-Test p-value:", bp_test_result[3])
```

### Checking for Normality
```python
# Histogram of normalized residuals
plt.figure(figsize=(8, 5))
plt.hist(fitted.resid_pearson, bins=8, edgecolor='k')
plt.ylabel('Count', fontsize=15)
plt.xlabel('Normalized residuals', fontsize=15)
plt.title("Histogram of normalized residuals", fontsize=18)
plt.show()

# Q-Q plot of the residuals
from statsmodels.graphics.gofplots import qqplot
plt.figure(figsize=(8, 5))
fig = qqplot(fitted.resid_pearson, line='45', fit=True)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("Theoretical quantiles", fontsize=15)
plt.ylabel("Sample quantiles", fontsize=15)
plt.title("Q-Q plot of normalized residuals", fontsize=18)
plt.grid(True)
plt.show()
```

### Checking for Outliers
```python
from statsmodels.stats.outliers_influence import OLSInfluence as influence

# Plotting Cook's distance
inf = influence(fitted)
(c, p) = inf.cooks_distance
plt.figure(figsize=(8, 5))
plt.title("Cook's distance plot for the residuals", fontsize=16)
plt.plot(np.arange(len(c)), c, marker='o', linestyle='-')
plt.grid(True)
plt.show()
```

## Conclusion
This notebook serves as a comprehensive guide to advanced multiple linear regression analysis using the `mtcars` dataset. It includes thorough steps for fitting the model and diagnosing key assumptions, providing valuable insights and techniques for your regression analysis projects.

## Repository Structure
- **notebooks/**: Directory containing the Jupyter Notebook for the analysis.
- **data/**: Directory containing the dataset (if applicable).
- **README.md**: Overview of the repository and instructions.

Feel free to explore, fork, and contribute to this repository!

