# Smart Finance Analyzer

A web application that automatically categorizes personal finance transactions, detects unusual spending behavior, and forecasts next month's expenses using Machine Learning.

---

## Live Demo

Live App: https://smart-finance-analyzer-raajvir15.streamlit.app

---

## Screenshot

![App Screenshot](assets/screenshot1.png)
![App Screenshot](assets/screenshot2.png)
![App Screenshot](assets/screenshot3.png)
![App Screenshot](assets/screenshot4.png)
![App Screenshot](assets/screenshot5.png)
![App Screenshot](assets/screenshot6.png)
---

## Features

### Automatic Transaction Categorization

* Classifies transactions into categories such as Food, Rent, Utilities, Shopping, Transport, Entertainment, Healthcare, and more.
* Uses a machine learning pipeline combining TF-IDF text features with transaction amount and payment mode.
* Achieved 87.1% accuracy on unseen test data.

### Anomaly Detection

* Identifies suspicious or unusually large transactions within each spending category.
* Uses a category-wise Z-Score based anomaly detection approach.
* Highlights transactions that significantly deviate from normal spending behavior.

### Expense Forecasting

* Predicts next month's spending for each category.
* Provides lower and upper confidence bounds to indicate uncertainty.
* Helps users estimate future budgets and spending patterns.

### Interactive Dashboard

* Displays categorized transactions in a searchable table.
* Visualizes spending breakdown by category using bar and pie charts.
* Shows monthly spending trends over time.
* Presents anomaly alerts and spending forecasts in a single interface.

---

## Tech Stack

| Library      | Purpose                          |
| ------------ | -------------------------------- |
| scikit-learn | Transaction categorization model |
| Prophet      | Time series forecasting          |
| Streamlit    | Web application                  |
| Plotly       | Interactive visualizations       |
| Pandas       | Data manipulation and analysis   |
| NumPy        | Numerical computations           |
| Joblib       | Model serialization and loading  |

---

## Project Structure

```text
smart_finance_analyzer/
├── notebooks/
│   ├── 1_eda.ipynb                   # Exploratory data analysis
│   ├── 2_categorization_model.ipynb     # Categorization based on description only
│   ├── 3_categorization_model_v2.ipynb     # Multi-feature categorization model
│   ├── 4_anomaly_detection.ipynb     # Z-score vs Isolation Forest comparison
│   └── 5_forecasting.ipynb           # Prophet forecasting 
│
├── data/
│   └── sample_transactions.xlsx              # dataset used
│
├── models/
│   ├── categorizer_v2_multifeature.pkl
│   ├── anomaly_category_stats.pkl
│   ├── anomaly_threshold.pkl
│   └── forecasts.pkl
│
├── images/
│   └── icon.png                      
│
├── assets/
│   ├── screenshot1.png
|   ├── screenshot2.png
│   ├── screenshot2.png
│   ├── screenshot4.png
│   ├── screenshot5.png
│   ├── screenshot6.png            
│
├── requirements.txt                  # dependencies
└── app.py                            # Streamlit application
```

---

## How To Run Locally

```bash
git clone https://github.com/raajvir15/finance_analyzer_predictor

cd smart-finance-analyzer

python -m venv venv

# Windows
.\venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

---

## Model Performance

### Categorization Model (v2)

* Algorithm: Random Forest + TF-IDF + StandardScaler + OneHotEncoder
* Features:

  * Transaction Description
  * Transaction Amount
  * Payment Mode
* Accuracy: 87.1%
* Improvements made:

  * Rent category F1 Score improved from 0.50 to 0.96
  * Significant improvement achieved through feature engineering and inclusion of structured transaction information

### Anomaly Detection

#### Z-Score Method (Selected)

* Threshold: 2.0
* Precision: 1.00
* Recall: 0.69
* F1 Score: 0.82

#### Isolation Forest

* Precision: 0.38
* Recall: 0.38
* F1 Score: 0.38

Production Choice: Z-Score was selected due to substantially higher precision and overall F1 performance.

### Forecasting

* Algorithm: Prophet
* Forecast Horizon: Next Month
* Outputs:

  * Predicted Spending
  * Lower Confidence Bound
  * Upper Confidence Bound

---

## Key Learnings and Limitations

This project demonstrated how combining text features with structured transaction data can significantly improve classification performance. Feature engineering played a larger role in accuracy gains than model complexity alone.

The forecasting component would benefit from at least 24 months of historical transaction data for greater reliability and seasonality detection.

The dataset used in this project is synthetic. Real-world bank transaction exports contain noisier descriptions, abbreviations, merchant codes, and missing values, which would require additional preprocessing.

Category ambiguity remains the primary source of misclassification, particularly when transaction descriptions contain limited contextual information.

---

## Future Improvements

* Add support for direct bank statement imports.
* Deploy a real-time transaction monitoring pipeline.
* Introduce personalized budgeting recommendations.
* Replace rule-based anomaly thresholds with adaptive user-specific models.
* Add user authentication and historical report storage.

---

## Author

Developed as an end-to-end Machine Learning project demonstrating:

* NLP-based transaction categorization
* Anomaly detection
* Time series forecasting
* Interactive dashboard development
* Model deployment with Streamlit
