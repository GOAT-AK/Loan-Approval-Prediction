# Loan-Approval-Prediction
This project focuses on predicting whether a loan application will be approved or rejected based on multiple financial and personal factors. The model is built using Logistic Regression with hyperparameter tuning (GridSearchCV) and deployed with Streamlit for real-time predictions.

# Table of Content

* [Brief](#Brief)  
* [DataSet](#DataSet)  
* [How_It_Works](#How_It_Works)  
* [Tools](#Tools)
* [Model_Performance](#Model_Performance)  
* [Remarks](#Remarks)  
* [Usage](#Usage)  
* [Sample_Run](#Sample_Run)


# Brief

With the growing need for automated financial risk assessment, predicting loan approval is a critical task for lending institutions.
This project builds a machine learning classification model that uses applicants’ financial records, credit history, and asset values to determine whether their loan will be approved (1) or rejected (0).



# DataSet

The dataset used in this project is the [Loan-Approval-Prediction-Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) from Kaggle. It includes key applicant information such as ***income, dependents, CIBIL score, loan amount, term, and asset values.***


### Column Descriptions

| Attribute                           | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| loan_id                             | Unique loan identifier.                                                     |
| no_of_dependents                    | Number of dependents of the applicant.                                      |
| education                           | Education level of the applicant.                                           |
| self_employed                       | Employment status of the applicant.                                         |
| income_annum                        | Annual income of the applicant.                                             |
| loan_amount                         | Loan amount requested.                                                      |
| loan_term                           | Loan term in years.                                                         |
| cibil_score                         | Applicant’s credit score.                                                   |
| residential_assets_value            | Value of residential assets.                                                |
| commercial_assets_value             | Value of commercial assets.                                                 |
| luxury_assets_value                 | Value of luxury assets.                                                     |
| bank_asset_value                    | Value of bank assets.                                                       |
| loan_status                         | Loan approval status (Target: 0 = Rejected, 1 = Approved).                  |    


# How_It_Works

- Load and clean the dataset (handle missing values, strip column names, remove unnecessary features). 
- Perform **Exploratory Data Analysis (EDA)** and run **ANOVA tests** to check feature importance.
- Apply ***SMOTE*** to balance the dataset (handle loan approval imbalance).  
- Scale numerical features using ***StandardScaler***.  
- Train a ***Logistic Regression model*** with ***GridSearchCV*** to tune hyperparameters (`C`, `penalty`, `solver`, `class_weight`).  
- Evaluate the model using ***accuracy, precision, recall, and F1-score***.
- Save the trained model, scaler, and feature list using ***pickle***.
- Deploy a ***Streamlit app*** that allows users to input loan application details and get predictions instantly.   


# Tools & Libraries

I. Jupyter Notebook & VS Code  
II. Python 3.x  
III. pandas, numpy  
IV. matplotlib, seaborn  
V. scikit-learn  
VI. imbalanced-learn (SMOTE)  
VII. pickle  
VIII. Streamlit 


# Model_Performance

The Logistic Regression model was tuned and evaluated on the test set.  
Evaluation results showed:  

- **Accuracy:** 93.78%  
- **Classification Report:**  

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Rejected) | 0.97 | 0.93 | 0.95 | 521 |
| 1 (Approved) | 0.89 | 0.96 | 0.92 | 315 |
| **Overall Accuracy** | **0.94** | | | 836 |
| **Macro Avg** | 0.93 | 0.94 | 0.93 | 836 |
| **Weighted Avg** | 0.94 | 0.94 | 0.94 | 836 |

This indicates that the model can generalize well and provide meaningful predictions on new loan applications.  



# Remarks
* This Python program was run and tested in Jupyter Notebook.
* Ensure the required libraries are installed by running:

  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn streamlit

# Usage

To begin utilizing this application, follow these steps:

1. Clone this repository:
   
   ```bash
   git clone https://github.com/GOAT-AK/Loan-Approval-Prediction

2. Navigate to the cloned repository:

   ```bash
   cd Loan-Approval-Prediction

3. Run the Jupyter Notebook:

   ```bash
   Loan Approval.ipynb

4. Launch the Streamlit app:
   
   ```bash
   streamlit run Script_loan.py

   
# Sample_Run


* Pic 1

<img width="1131" height="718" alt="Image" src="https://github.com/user-attachments/assets/438dfda7-aef0-446d-a181-aaff233764a6" />



<hr>


* Pic 2

<img width="1131" height="662" alt="Image" src="https://github.com/user-attachments/assets/a74d3cc0-3e2c-4df1-9746-63cfa38d397d" />
   

