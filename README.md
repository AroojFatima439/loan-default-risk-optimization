# Loan Default Risk with Business Cost Optimization

## 📌 Objective
Predict the likelihood of a loan default using binary classification models and optimize the classification threshold based on a business cost-benefit analysis.

---

## 📂 Dataset Description

The dataset contains customer-level financial information with the following features:

- `person_age`: Age of the borrower
- `person_income`: Annual income
- `person_home_ownership`: Type of home ownership
- `person_emp_length`: Length of employment in years
- `loan_intent`: Purpose of the loan
- `loan_grade`: Loan grade assigned
- `loan_amnt`: Amount of loan taken
- `loan_int_rate`: Interest rate on the loan
- `loan_status`: Target variable (1 = Default, 0 = Paid)
- `loan_percent_income`: Loan amount as a % of income
- `cb_person_default_on_file`: Whether person had previous default (Y/N)
- `cb_person_cred_hist_length`: Credit history length in years

---

## 🛠️ Task Performed

### 1. **Data Preprocessing**
- Dropped missing values
- Encoded categorical variables using Label Encoding
- Normalized numerical features using `StandardScaler`
- Split dataset into training and testing sets (80/20)

### 2. **Model Training**
- Trained two binary classification models:
  - Logistic Regression
  - CatBoost Classifier (gradient boosting)

### 3. **Business Cost Modeling**
Defined custom cost values:
- **False Positive (FP)** → Cost: `1000`  
  *(Approved a loan to a defaulter)*
- **False Negative (FN)** → Cost: `500`  
  *(Rejected a good customer)*

Evaluated different probability thresholds (from 0.1 to 0.9) to:
- Predict class based on threshold
- Calculate confusion matrix
- Compute total business cost
- Find **optimal threshold** minimizing cost

---

## 📈 Results

- **Optimal Threshold**: `0.56`
- **Confusion Matrix**:  
  - TP = 873  
  - FP = 11  
  - TN = 4476  
  - FN = 368  
- **Total Business Cost**: `195,000`
- **ROC AUC Score**: `0.9428`

---

## 📊 Visualization

A graph was plotted to show **Total Business Cost vs Threshold**. The lowest point on the curve corresponds to the optimized decision threshold (0.56), balancing between false approvals and false rejections.

---

## 🧠 Skills Gained

- Binary classification and probability-based predictions  
- Business-aware model evaluation using cost metrics  
- Threshold tuning to minimize losses  
- Feature encoding, scaling, and model comparison  
- ROC AUC interpretation and confusion matrix analysis

---

## 🚀 How to Run

1. Install required packages:
```bash
pip install pandas numpy scikit-learn catboost matplotlib

