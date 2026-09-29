# Bank Customer Conversion Intelligence

### Predicting and Prioritizing Customers for Term Deposit Campaigns

An end-to-end Machine Learning project that analyzes bank customer data, predicts the likelihood of term deposit subscription, ranks customers by conversion probability, and provides business-oriented insights through an interactive Streamlit dashboard.

---

## 📌 Overview

Banks often contact large numbers of customers during marketing campaigns, but customers do not have the same likelihood of subscribing to a term deposit.

This project builds a machine learning solution that helps identify and prioritize customers who are more likely to subscribe.

The project covers the complete workflow:

**Data Quality → EDA → Preprocessing → Machine Learning → Model Selection → Threshold Optimization → Customer Ranking → SHAP Explainability → Business Targeting → Interactive Dashboard**

---

## 🎯 Business Problem

The bank wants to improve the efficiency of its term deposit marketing campaigns.

Instead of treating every customer equally, the goal is to:

* Identify customers with higher conversion probability.
* Prioritize customers for marketing campaigns.
* Understand customer and campaign-related patterns.
* Reduce unnecessary targeting of low-probability customers.
* Provide an interpretable prediction system for business users.

The prediction scenario is **before the campaign starts**, using only information available at that point.

---

## 🎯 Project Objectives

1. Analyze customer and campaign characteristics.
2. Investigate data quality and potential anomalies.
3. Build a pre-campaign prediction pipeline.
4. Compare multiple machine learning models.
5. Optimize the classification threshold.
6. Rank customers according to predicted conversion probability.
7. Analyze model behavior using SHAP.
8. Translate model predictions into business targeting insights.
9. Build an interactive Streamlit dashboard.
10. Provide a live customer prediction interface.

---

## 📊 Dataset

The dataset contains **45,211 customer records** and includes demographic, financial, contact, and previous campaign information.

### Target Variable

`y`

* `yes` → Customer subscribed to a term deposit.
* `no` → Customer did not subscribe.

The overall conversion rate is approximately:

**11.7%**

This makes the dataset imbalanced, which is why accuracy alone is not sufficient for evaluating the model.

### Main Features

| Feature     | Description                                    |
| ----------- | ---------------------------------------------- |
| `age`       | Customer age                                   |
| `job`       | Type of job                                    |
| `marital`   | Marital status                                 |
| `education` | Education level                                |
| `default`   | Has credit in default                          |
| `balance`   | Average yearly balance                         |
| `housing`   | Housing loan                                   |
| `loan`      | Personal loan                                  |
| `contact`   | Contact communication type                     |
| `day`       | Day of last contact                            |
| `month`     | Month of last contact                          |
| `duration`  | Duration of last contact                       |
| `campaign`  | Number of contacts during the current campaign |
| `pdays`     | Days since previous campaign contact           |
| `previous`  | Number of previous contacts                    |
| `poutcome`  | Outcome of previous campaign                   |

---

## 🔍 Data Quality & EDA

The dataset was investigated before modeling rather than directly feeding the data into machine learning algorithms.

### Data Quality Checks

* No missing values.
* No duplicate rows.
* `unknown` categories were investigated and retained where they represent meaningful missing information.
* Negative account balances were treated as valid financial values.
* `pdays = -1` was interpreted as customers who had not previously been contacted.
* Extreme values were investigated instead of being automatically removed.

### Important EDA Findings

Some customer groups showed substantially different subscription rates.

Examples include:

* Previous campaign success is strongly associated with subscription.
* Customers with previous campaign history showed higher conversion than customers with no previous contact.
* Conversion varies across job categories.
* Conversion changes substantially across months.
* Customers without housing or personal loans generally showed higher observed conversion rates.
* The relationship between age and conversion is nonlinear.
* Repeated contacts during the current campaign are associated with lower conversion rates.

These relationships describe patterns in the dataset and should not be interpreted as causal effects.

---

## ⚙️ Machine Learning Pipeline

The prediction scenario is designed to operate **before the current campaign begins**.

Therefore, the following features were excluded from the predictive model:

* `duration`
* `campaign`

These variables are not assumed to be available before the campaign starts.

The final feature set contains:

```text
age
job
marital
education
default
balance
housing
loan
contact
day
month
pdays
previous
poutcome
```

### Preprocessing

Different preprocessing strategies were applied depending on feature type.

* Numerical features → retained as numerical values.
* Binary categorical features → encoded.
* Ordinal education → ordinal encoding.
* Nominal categorical features → one-hot encoding.
* `month` → transformed into cyclical features using sine and cosine transformations.
* Unknown categories → retained.
* `OneHotEncoder(handle_unknown="ignore")` → used for robust handling of unseen categories.

The preprocessing pipeline was fitted only on the training data to avoid data leakage.

---

## 🤖 Models

Three machine learning approaches were evaluated:

### Logistic Regression

Used as a baseline linear classification model.

### Random Forest

Used to capture nonlinear relationships and feature interactions.

### XGBoost

Used as the final candidate because of its strong performance on the validation data and its ability to model nonlinear relationships.

---

## 🏆 Model Selection

Models were compared using metrics that are more informative for an imbalanced classification problem.

| Model               | Validation ROC-AUC | Validation PR-AUC | Best Validation F1 |
| ------------------- | -----------------: | ----------------: | -----------------: |
| Logistic Regression |              0.749 |                 — |              0.380 |
| Random Forest       |              0.804 |             0.456 |              0.493 |
| XGBoost             |          **0.811** |         **0.477** |          **0.504** |

XGBoost was selected based on the validation results.

The model was then evaluated on the held-out test set.

---

## 🎚️ Threshold Optimization

The default classification threshold of `0.5` was not used automatically.

Instead, the validation set was used to evaluate different probability thresholds.

The selected threshold was:

```text
0.22
```

At this threshold, the XGBoost model achieved its highest validation F1 among the evaluated thresholds.

This threshold was then fixed before evaluating the final model on the test set.

---

## 📈 Final Test Performance

The final XGBoost model was evaluated on an untouched test set.

| Metric    | Test Result |
| --------- | ----------: |
| ROC-AUC   |   **0.805** |
| PR-AUC    |   **0.460** |
| Precision |   **0.475** |
| Recall    |   **0.497** |
| F1 Score  |   **0.486** |
| Accuracy  |   **0.877** |

### Confusion Matrix

```text
                  Predicted
                 No      Yes
Actual No       7404     581
Actual Yes       532     526
```

The model correctly identified 526 subscribing customers while generating 581 false positives.

Because the positive class represents only around 11.7% of customers, accuracy by itself does not fully describe the usefulness of the model.

---

## 🎯 Customer Targeting

The model is not only used for binary classification.

Customers are also **ranked by their predicted probability of subscription**.

This allows the bank to focus marketing resources on higher-probability customers.

### Test Set Results

| Targeted Segment | Customers | Conversion Rate |      Lift |
| ---------------- | --------: | --------------: | --------: |
| Top 10%          |       904 |       **52.5%** | **4.49×** |
| Top 20%          |     1,808 |       **36.3%** | **3.11×** |
| Top 30%          |     2,712 |       **27.8%** | **2.38×** |
| Top 40%          |     3,617 |       **22.9%** | **1.96×** |
| Top 50%          |     4,521 |       **19.6%** | **1.67×** |
| All Customers    |     9,043 |       **11.7%** |     1.00× |

The top 10% of customers in the held-out test set had an observed conversion rate of approximately **52.5%**, compared with the overall test-set baseline of approximately **11.7%**.

This demonstrates that the model can be useful as a **customer prioritization and ranking system**, not only as a binary classifier.

> These results describe performance on the held-out test set and should not be interpreted as guaranteed future campaign performance.

---

## 🔎 Explainability with SHAP

SHAP was used to understand how the trained XGBoost model uses features when generating predictions.

The analysis helps answer questions such as:

* Which features have the largest influence on predictions?
* Which features contribute most to model variation?
* How does the model behavior differ across customers?

The most influential features included:

* `marital_divorced`
* `contact_cellular`
* `poutcome_unknown`
* `poutcome_success`
* `contact_unknown`
* `balance`
* `poutcome_failure`
* `job_unemployed`
* `job_student`
* `poutcome_other`

SHAP describes **model behavior rather than causality**. A feature having a large SHAP value does not mean that changing that feature would necessarily cause a customer to subscribe.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard with four main sections.

### 1. Overview

Provides a high-level view of:

* Customer population.
* Conversion rate.
* Model performance.
* Key business metrics.

### 2. Model Performance

Displays:

* Model comparison.
* ROC-AUC.
* PR-AUC.
* Precision.
* Recall.
* F1 Score.
* Confusion Matrix.
* Classification threshold.

### 3. Customer Targeting

Provides:

* Customer ranking.
* Probability-based targeting.
* Decile analysis.
* Conversion rate.
* Lift.
* Gain analysis.

### 4. Live Prediction

Allows users to enter customer information and generate an individual prediction.

The page provides:

* Predicted subscription probability.
* Classification based on the selected threshold.
* Customer targeting interpretation.

---

## 📁 Project Structure

```text
Bank-Customer-Conversion-Intelligence/
│
├── app/
│   └── app.py
│
├── data/
│   └── bank_marketing.csv
│
├── models/
│   ├── xgb_model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   ├── 01_Data_Quality_and_EDA.ipynb
│   ├── 02_Preprocessing_and_Modeling.ipynb
│   └── 03_SHAP_and_Business_Analysis.ipynb
│
├── outputs/
│   └── test_results.csv
│
├── screenshots/
│   ├── overview.png
│   ├── model_performance.png
│   ├── customer_targeting.png
│   └── live_prediction.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Bank-Customer-Conversion-Intelligence
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Dashboard

From the project root:

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

## 💡 Key Business Insights

The project demonstrates several important findings:

### Customer ranking can be more useful than binary classification

Rather than simply predicting `yes/no`, the model can rank customers according to their predicted probability.

### Previous campaign history is informative

Customers with previous campaign interactions showed substantially different subscription behavior compared with customers without previous contact.

### Customer behavior is not uniform

Conversion rates vary across demographic, financial, contact, and historical campaign characteristics.

### Campaign resources can be prioritized

The ranking results allow a business to focus its resources on a smaller group of higher-probability customers instead of contacting every customer equally.

---

## ⚠️ Limitations

* The dataset represents historical campaign behavior.
* Observed relationships should not be interpreted as causal effects.
* Model performance on historical data does not guarantee future campaign performance.
* The threshold of `0.22` was selected based on validation F1 and may not be optimal for every business scenario.
* A production system should incorporate the actual cost of contacting customers and the value of successful conversions.
* Probability calibration should be evaluated before treating predicted probabilities as reliable business probabilities.
* Real-world deployment would require monitoring for data drift and model performance degradation.

---

## 🔮 Future Improvements

Potential extensions include:

* Probability calibration.
* Cost-sensitive threshold optimization.
* Campaign ROI optimization.
* Model monitoring.
* Data drift detection.
* Automated retraining.
* A production database connection.
* API-based model serving.
* A/B testing of targeting strategies.
* Real-time campaign feedback.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* Jupyter Notebook
* Joblib

---

## 👨‍💻 Project Focus

This project demonstrates an end-to-end Machine Learning workflow with an emphasis on:

**Data Quality → Machine Learning → Model Evaluation → Explainability → Customer Ranking → Business Decision Support**

It goes beyond building a classification model by connecting model predictions to a practical customer-targeting workflow.
