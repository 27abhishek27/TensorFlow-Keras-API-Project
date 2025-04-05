# TensorFlow Keras API Project

## 📌 Overview

This project utilizes the **TensorFlow Keras API** to build and evaluate deep learning models for **loan default prediction** using data from LendingClub. The objective is to predict whether a borrower will default on a loan based on various financial and demographic features.

## 📂 Project Structure

```
├── TensorFlow Keras Project Png/  # Visualizations and plots generated during analysis
├── lending_club_info.csv          # Metadata describing the dataset features
├── lending_club_loan_two.csv      # Main dataset containing loan records
├── TensorFlow-Keras-Project.ipynb # Jupyter Notebook with data analysis and model implementation
├── README.md                      # Project documentation
└── .gitattributes                  # Git configuration attributes
```

## 📊 Dataset

The dataset used in this project is sourced from **LendingClub**, comprising loan records with the following key features:

- **loan_amnt**: The loan amount requested by the borrower.
- **term**: The duration of the loan (e.g., 36 or 60 months).
- **int_rate**: The interest rate assigned to the loan.
- **installment**: The fixed monthly payment owed by the borrower.
- **grade**: LendingClub's assigned loan grade.
- **sub_grade**: A more granular loan grade.
- **emp_title**: The job title of the borrower.
- **emp_length**: The length of the borrower's employment.
- **home_ownership**: The borrower's home ownership status.
- **annual_inc**: The annual income of the borrower.
- **verification_status**: Indicates if the income was verified.
- **purpose**: The purpose of the loan (e.g., debt consolidation, credit card).
- **title**: The loan title provided by the borrower.
- **zip_code**: The first three digits of the borrower's zip code.
- **addr_state**: The state provided by the borrower in the loan application.
- **dti**: Debt-to-income ratio.
- **earliest_cr_line**: The date when the borrower's earliest reported credit line was opened.
- **open_acc**: The number of open credit lines in the borrower's credit file.
- **pub_rec**: The number of derogatory public records.
- **revol_bal**: The borrower's revolving balance.
- **revol_util**: The revolving line utilization rate.
- **total_acc**: The total number of credit lines currently in the borrower's credit file.
- **initial_list_status**: The initial listing status of the loan.
- **application_type**: Indicates whether the loan is an individual or joint application.
- **mort_acc**: The number of mortgage accounts.
- **pub_rec_bankruptcies**: The number of public record bankruptcies.
- **loan_status**: The target variable indicating if the loan was fully paid or charged off.

## 🚀 Installation

### 1️⃣ Clone the repository:

```bash
git clone https://github.com/27abhishek27/TensorFlow-Keras-API-Project.git
cd TensorFlow-Keras-API-Project
```

### 2️⃣ Install dependencies:

Ensure you have the following Python packages installed:

- `tensorflow`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`

You can install them using pip:

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

## 🔍 Methodology

### 1. **Data Preprocessing**

- **Handling Missing Values**: Identified and addressed missing data in columns like `emp_title`, `emp_length`, `title`, `mort_acc`, and `revol_util`.
- **Feature Engineering**: Created new features and transformed existing ones, such as encoding categorical variables (`grade`, `sub_grade`, `home_ownership`, etc.) and extracting information from date fields.
- **Data Cleaning**: Removed or adjusted irrelevant or redundant features to improve model performance.

### 2. **Exploratory Data Analysis (EDA)**

- **Visualized distributions** of key features to understand data patterns.
- **Correlation Analysis**: Examined relationships between features and the target variable (`loan_status`).
- **Handled Imbalanced Data**: Addressed any class imbalances in the target variable.

### 3. **Model Building**

- **Neural Network Architecture**: Designed a deep learning model using TensorFlow's Keras API with multiple layers, including input, hidden, and output layers.
- **Activation Functions**: Utilized activation functions like ReLU for hidden layers and sigmoid for the output layer.
- **Compilation**: Compiled the model with appropriate loss functions and optimizers suitable for binary classification.

### 4. **Model Training and Evaluation**

- **Data Splitting**: Divided the dataset into training and testing sets.
- **Training**: Trained the neural network on the training data with techniques like early stopping to prevent overfitting.
- **Evaluation Metrics**: Assessed model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
- **Confusion Matrix**: Analyzed true positives, true negatives, false positives, and false negatives to evaluate classification performance.

## 📊 Visualizations

Here are some visualizations from the project:

![alt text](https://github.com/27abhishek27/TensorFlow-Keras-API-Project/blob/main/TensorFlow%20Keras%20Project%20Png/Create%20a%20bar%20plot%20showing%20the%20correlation%20of%20the%20numeric%20features%20to%20the%20new%20loan_repaid%20column.png)
![alt text](https://github.com/27abhishek27/TensorFlow-Keras-API-Project/blob/main/TensorFlow%20Keras%20Project%20Png/Display%20a%20count%20plot%20per%20subgrade%20with%20loan_status.png)
![alt text](https://github.com/27abhishek27/TensorFlow-Keras-API-Project/blob/main/TensorFlow%20Keras%20Project%20Png/It%20looks%20like%20F%20and%20G%20subgrades%20don't%20get%20paid%20back%20that%20often.png)
![alt text](https://github.com/27abhishek27/TensorFlow-Keras-API-Project/blob/main/TensorFlow%20Keras%20Project%20Png/Scatterplot.png)
![alt text](https://github.com/27abhishek27/TensorFlow-Keras-API-Project/blob/main/TensorFlow%20Keras%20Project%20Png/Visualize%20this%20using%20a%20heatmap.png)
![alt text](https://github.com/27abhishek27/TensorFlow-Keras-API-Project/blob/main/TensorFlow%20Keras%20Project%20Png/we%20will%20be%20attempting%20to%20predict%20loan_status%2C%20create%20a%20countplot.png)

## 🛠️ Technologies Used

- **Python**
- **TensorFlow & Keras**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**

## 📌 Future Improvements

- **Hyperparameter Tuning**: Implement techniques like GridSearchCV to optimize model parameters.
- **Feature Selection**: Explore advanced methods to select the most predictive features.
- **Model Deployment**: Develop a web interface to deploy the model for real-time loan default prediction.
- **Explainability**: Utilize tools like SHAP to interpret model predictions and understand feature importance.
