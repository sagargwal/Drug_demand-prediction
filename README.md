# Drug_demand prediction
This is a analysis I did in a hackathon hosted by HCL
# HCL Hackathon

## 📌 Project Overview
📊 Predicting Drug Demand with Time Series Forecasting & NLP! 💊📉
How can we predict drug demand more accurately? At the HCL Hackathon, I developed a time series forecasting model enhanced with Natural Language Processing (NLP) techniques to analyze market sentiment, helping optimize pharmaceutical supply chains and prevent shortages. 🚀

🔍 What We Did
📌 Data Cleaning & Preprocessing:

Handled missing values, outliers, and ensured stationarity for reliable forecasting.
Applied differencing and seasonal decomposition to identify trends and seasonality.
📌 Feature Engineering & NLP Integration:

Extracted time-dependent features like lag variables, moving averages, and trend indicators.
Used NLP-based sentiment analysis on customer reviews, social media discussions, and market reports to capture external demand drivers.
Converted text-based insights into quantitative sentiment scores to enhance forecasting models.
📌 Model Selection & Forecasting:

Applied traditional time series models: ARIMA, SARIMA, and Exponential Smoothing.
Used machine learning algorithms: XGBoost and Random Forest Regressor for capturing complex patterns.
Integrated sentiment analysis features into regression-based forecasting models to improve predictions.
📌 Model Evaluation & Insights:

Assessed performance using Mean Squared Error (MSE) and RMSE.
Identified correlation between sentiment trends and drug demand fluctuations.
Developed interactive visualizations for stakeholders to understand demand trends.
🏆 Key Takeaways
✅ Combining time series forecasting with NLP improves demand prediction accuracy.
✅ Market sentiment (customer reviews, news, and discussions) influences pharmaceutical demand.
✅ Feature engineering (lag values, moving averages, and text-derived features) plays a crucial role in forecasting.
✅ ML models like XGBoost outperform traditional ARIMA in certain cases when additional features are incorporated.

## 🚀 Getting Started

### Prerequisites
To run this notebook, install the required dependencies:

```bash
pip install -r requirements.txt
```

### Required Libraries
This project uses the following Python libraries:
```python
from math import sqrt
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import f_oneway
from scipy.stats import skew
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder  # Correct import
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import nltk
import numpy as np
import numpy as np 
import pandas as pd
import pmdarima as pm
import scipy.stats as stats
import seaborn as sns
import sklearn  
import xgboost as xgb
```

## 🔧 How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "hcl hackathon.ipynb"
   ```
3. Execute the cells sequentially.

## 📊 Dataset
Ensure the required dataset is available before running the notebook. If any external dataset is needed, update the `data/` directory accordingly.

## 🏆 Results & Insights
The notebook provides an end-to-end solution for the hackathon problem statement. Key findings and visualizations are presented at the end of the notebook.

## 🤝 Contributing
Feel free to fork the repo and submit a pull request.

## 📜 License
This project is licensed under the MIT License.

---

🛠 Developed for HCL Hackathon 🚀
