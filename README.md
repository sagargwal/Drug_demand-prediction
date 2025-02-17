# Drug_demand prediction
This is a analysis I did in a hackathon hosted by HCL
# HCL Hackathon

## 📌 Project Overview
No description available.

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
