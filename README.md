# 🩺 Cancer Diagnosis Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on developing a Machine Learning-based Cancer Diagnosis Prediction model using patient-related data.

The main objective is to analyze patient characteristics, identify patterns and relationships between input features and cancer diagnosis, and develop classification models to predict whether a patient belongs to the No Cancer or Cancer category.

This project follows an end-to-end Machine Learning workflow, including:

- Data Understanding
- Data Quality Analysis
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Data Preprocessing
- Machine Learning Model Development
- Model Evaluation
- Hyperparameter Tuning
- Model Comparison
- Feature Importance Analysis
- Model Saving and Loading

**Selected Model:** Tuned Decision Tree Classifier

**Final Test Accuracy:** 89.33%

---

## 🎯 Project Objectives

- Understand the dataset structure and features.
- Perform data quality analysis.
- Identify missing values, duplicate records, and potential outliers.
- Explore numerical and categorical features.
- Analyze relationships between features and the target variable.
- Perform statistical tests to identify significant associations.
- Apply appropriate data preprocessing techniques.
- Build classification models using Logistic Regression and Decision Tree.
- Evaluate models using multiple performance metrics.
- Optimize model performance using GridSearchCV.
- Compare original and tuned models.
- Select and save the preferred model.

---

## 📂 Dataset Information

**Dataset Name:** `The_Cancer_data_1500_V2.csv`

| Description | Details |
|---|---:|
| Total Records | 1,500 |
| Total Columns | 9 |
| Input Features | 8 |
| Target Variable | Diagnosis |
| Problem Type | Binary Classification |

### Features Description

| Feature | Description |
|---|---|
| Age | Patient age |
| Gender | Gender category (encoded) |
| BMI | Body Mass Index |
| Smoking | Smoking category (encoded) |
| GeneticRisk | Genetic risk category (encoded) |
| PhysicalActivity | Physical activity measurement |
| AlcoholIntake | Alcohol intake measurement |
| CancerHistory | Cancer history category (encoded) |
| Diagnosis | Target variable |

### Target Variable

- `0` – No Cancer
- `1` – Cancer

*Note: The original meanings of encoded categorical values and measurement units should be interpreted according to the dataset documentation.*

---

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Development environment |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning and preprocessing |
| Statsmodels | Statistical analysis and VIF |
| Joblib | Model saving and loading |
| GitHub | Project documentation and version control |

---

# 🔍 Project Workflow

## 1. Data Loading and Understanding

The dataset was loaded into a Pandas DataFrame.

Initial inspection was performed using:

- `head()`
- `tail()`
- `shape`
- `info()`
- `describe()`
- `columns`
- `dtypes`

The dataset contained 1,500 records and 9 columns, including 8 independent features and 1 target variable.

## 2. Data Quality Analysis

Several data quality checks were performed to understand the quality and structure of the dataset.

### 2.1 Missing Value Analysis

Missing values were checked across all columns.

**Result:** No missing values were identified.

### 2.2 Duplicate Records Analysis

Duplicate records were checked across the dataset.

**Result:** No duplicate rows were identified.

### 2.3 Data Type Analysis

The data types of all columns were examined.

- Integer features
- Floating-point features
- Target variable

### 2.4 Unique Value Analysis

Unique values and unique counts were examined for categorical and numerical features.

Categorical features included:

- Gender
- Smoking
- GeneticRisk
- CancerHistory
- Diagnosis

No unexpected encoded category values were identified in the inspected columns.

### 2.5 Outlier Detection Using IQR

The Interquartile Range (IQR) method was used to identify potential outliers in numerical features.

Numerical features examined:

- Age
- BMI
- PhysicalActivity
- AlcoholIntake

**Result:** No outliers were detected using the calculated IQR boundaries.

No records were removed during this data quality analysis.

---

## 3. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand feature distributions, patterns, and relationships within the dataset.

### 3.1 Descriptive Statistical Analysis

The following statistical measures were calculated:

- Mean
- Median
- Minimum
- Maximum
- Standard deviation
- Quartiles

### 3.2 Numerical Feature Distribution

Histograms were used to visualize the distributions of numerical features.

### 3.3 Skewness Analysis

Skewness was calculated to understand the symmetry of numerical feature distributions.

| Feature | Skewness |
|---|---:|
| Age | -0.0328 |
| BMI | -0.0214 |
| PhysicalActivity | 0.0760 |
| AlcoholIntake | 0.0551 |

The skewness values were close to zero, indicating approximately symmetric distributions.

### 3.4 Categorical Feature Analysis

Frequency distributions were examined for categorical features.

| Feature | Category | Count |
|---|---:|---:|
| Gender | 0 | 764 |
| Gender | 1 | 736 |
| Smoking | 0 | 1096 |
| Smoking | 1 | 404 |
| GeneticRisk | 0 | 895 |
| GeneticRisk | 1 | 447 |
| GeneticRisk | 2 | 158 |
| CancerHistory | 0 | 1284 |
| CancerHistory | 1 | 216 |

### 3.5 Target Variable Distribution

The target variable was analyzed to understand the distribution of the two classes.

| Diagnosis | Category | Count | Percentage |
|---|---|---:|---:|
| 0 | No Cancer | 943 | 62.87% |
| 1 | Cancer | 557 | 37.13% |
| **Total** | | **1500** | **100%** |

The dataset showed a moderate class imbalance.

### 3.6 Bivariate Analysis

Relationships between individual features and the target variable were explored.

Techniques included:

- Boxplots for numerical features against Diagnosis.
- Cross-tabulation for categorical features.
- Row-wise percentage analysis.

The analysis showed differences in numerical feature distributions between the two diagnosis categories.

The Cancer category showed higher median values for Age, BMI, and AlcoholIntake, and a lower median value for PhysicalActivity in the examined boxplots.

### 3.7 Correlation Analysis

Pearson correlation analysis was performed to examine relationships between numerical features and the target variable.

| Feature | Correlation with Diagnosis |
|---|---:|
| Age | 0.1966 |
| BMI | 0.1876 |
| PhysicalActivity | -0.1501 |
| AlcoholIntake | 0.2128 |

The numerical features showed relatively weak individual linear correlations with the target variable.

A correlation heatmap was also used for visualization.

### 3.8 Chi-Square Test of Independence

The Chi-Square test was performed to investigate associations between categorical features and Diagnosis.

| Feature | Chi-Square Statistic | Degrees of Freedom |
|---|---:|---:|
| Gender | 92.9685 | 1 |
| Smoking | 76.2373 | 1 |
| GeneticRisk | 167.4326 | 2 |
| CancerHistory | 228.4108 | 1 |

All four tests produced p-values below 0.05.

This indicates statistically significant associations between the examined categorical features and Diagnosis in this dataset.

### 3.9 Cramér's V Analysis

Cramér's V was calculated to measure the strength of association between categorical features and Diagnosis.

| Feature | Cramér's V | Association Strength |
|---|---:|---|
| Gender | 0.2490 | Weak |
| Smoking | 0.2254 | Weak |
| GeneticRisk | 0.3341 | Moderate |
| CancerHistory | 0.3902 | Moderate |

CancerHistory showed the strongest categorical association among the examined features.

### 3.10 Multicollinearity Analysis – VIF

Variance Inflation Factor (VIF) was calculated to investigate multicollinearity among numerical features.

| Feature | VIF |
|---|---:|
| Age | 1.0012 |
| BMI | 1.0011 |
| PhysicalActivity | 1.0015 |
| AlcoholIntake | 1.0012 |

The VIF values were close to 1, indicating no significant multicollinearity among the examined numerical features.

---

# ⚙️ 4. Data Preprocessing

Data preprocessing was performed before developing the Machine Learning models.

### 4.1 Feature and Target Separation

Independent variables:

`X = df.drop('Diagnosis', axis=1)`

Dependent variable:

`y = df['Diagnosis']`

Shapes:

- X Shape: `(1500, 8)`
- y Shape: `(1500,)`

### 4.2 Train-Test Split

The dataset was divided into training and testing sets.

| Dataset | Records |
|---|---:|
| Training Set | 1200 |
| Testing Set | 300 |
| Total | 1500 |

Configuration:

- Test size: 20%
- Training size: 80%
- Random state: 42
- Stratified sampling: Enabled

### 4.3 Feature Scaling

`StandardScaler` was used to standardize numerical features.

Numerical features:

- Age
- BMI
- PhysicalActivity
- AlcoholIntake

### 4.4 Categorical Encoding

`OneHotEncoder` was applied to categorical features.

Categorical features:

- Gender
- Smoking
- GeneticRisk
- CancerHistory

`handle_unknown='ignore'` was used to handle previously unseen categories.

### 4.5 ColumnTransformer

`ColumnTransformer` was used to apply different preprocessing techniques to numerical and categorical features.

The preprocessing process produced:

- 4 numerical features
- 9 encoded categorical features
- 13 transformed features in total

The preprocessing pipeline was incorporated into the hyperparameter tuning pipelines so that preprocessing was fitted within each cross-validation training fold.

---

# 🤖 5. Machine Learning Model Development

Two classification algorithms were initially developed and evaluated.

## 5.1 Logistic Regression

Logistic Regression was implemented as the first classification model.

### Original Logistic Regression Results

| Metric | Score |
|---|---:|
| Accuracy | 88.00% |
| Precision | 84.40% |
| Recall | 82.88% |
| F1-score | 83.64% |
| ROC-AUC | 94.10% |

**Confusion Matrix:**

```text
[[172  17]
 [ 19  92]]
```

## 5.2 Decision Tree Classifier

A Decision Tree Classifier was implemented as the second classification model.

### Original Decision Tree Results

| Metric | Score |
|---|---:|
| Accuracy | 87.67% |
| Precision | 81.36% |
| Recall | 86.49% |
| F1-score | 83.84% |
| ROC-AUC | 87.42% |

**Confusion Matrix:**

```text
[[167  22]
 [ 15  96]]
```

---

# 🔧 6. Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used to optimize both classification models.

Recall was selected as the hyperparameter optimization scoring metric.

## 6.1 Decision Tree Hyperparameter Tuning

Parameters explored:

| Parameter | Values |
|---|---|
| Criterion | Gini, Entropy |
| Maximum Depth | None, 3, 5, 7, 10 |
| Minimum Samples Split | 2, 5, 10 |
| Minimum Samples Leaf | 1, 2, 5 |

- Total parameter combinations: 90
- Cross-validation folds: 5
- Total fits: 450

### Best Parameters

```python
{
    'classifier__criterion': 'entropy',
    'classifier__max_depth': None,
    'classifier__min_samples_leaf': 5,
    'classifier__min_samples_split': 2
}
```

**Best Cross-Validation Recall:** 78.92%

### Tuned Decision Tree Test Results

| Metric | Score |
|---|---:|
| Accuracy | 89.33% |
| Precision | 83.76% |
| Recall | 88.29% |
| F1-score | 85.96% |
| ROC-AUC | 92.30% |

**Confusion Matrix:**

```text
[[170  19]
 [ 13  98]]
```

## 6.2 Logistic Regression Hyperparameter Tuning

GridSearchCV was used to optimize:

| Parameter | Values |
|---|---|
| Regularization parameter (C) | 0.01, 0.1, 1, 10, 100 |
| Class Weight | None, Balanced |

- Total parameter combinations: 10
- Cross-validation folds: 5
- Total fits: 50

### Best Parameters

```python
{
    'classifier__C': 0.1,
    'classifier__class_weight': 'balanced'
}
```

**Best Cross-Validation Recall:** 86.77%

### Tuned Logistic Regression Test Results

| Metric | Score |
|---|---:|
| Accuracy | 87.00% |
| Precision | 80.00% |
| Recall | 86.49% |
| F1-score | 83.12% |
| ROC-AUC | 93.91% |

**Confusion Matrix:**

```text
[[165  24]
 [ 15  96]]
```

---

# 📊 7. Model Comparison

All four models were compared using the same test dataset.

| Evaluation Metric | Original LR | Tuned LR | Original DT | Tuned DT |
|---|---:|---:|---:|---:|
| Accuracy | 88.00% | 87.00% | 87.67% | **89.33%** |
| Precision | **84.40%** | 80.00% | 81.36% | 83.76% |
| Recall | 82.88% | 86.49% | 86.49% | **88.29%** |
| F1-score | 83.64% | 83.12% | 83.84% | **85.96%** |
| ROC-AUC | **94.10%** | 93.91% | 87.42% | 92.30% |
| False Negatives | 19 | 15 | 15 | **13** |

### ROC Curve Comparison

ROC curves were plotted for all four models to compare their classification performance across different probability thresholds.

Results:

- Original Logistic Regression: 0.941
- Tuned Logistic Regression: 0.939
- Original Decision Tree: 0.874
- Tuned Decision Tree: 0.923

Original Logistic Regression achieved the highest ROC-AUC, while Tuned Decision Tree achieved the highest Recall and F1-score.

### Confusion Matrix Comparison

Confusion matrices were visualized to compare:

- True Positives
- True Negatives
- False Positives
- False Negatives

The Tuned Decision Tree achieved the lowest False Negative count, with 13 records.

---

# 🏆 8. Final Model Selection

**Selected Model: Tuned Decision Tree Classifier**

The Tuned Decision Tree Classifier was selected as the preferred model based on the overall test evaluation.

Reasons for selection:

- Highest test accuracy: 89.33%.
- Highest test recall: 88.29%.
- Highest test F1-score: 85.96%.
- Lowest false negative count: 13.
- Improved performance compared with the original Decision Tree model.

Although Original Logistic Regression achieved the highest ROC-AUC of 0.941, Tuned Decision Tree provided better results for the selected threshold-based metrics.

Therefore, Tuned Decision Tree was selected as the final candidate model for this educational project.

---

# 🔎 9. Feature Importance Analysis

Feature importance was extracted from the selected Tuned Decision Tree model.

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Age | 0.153392 |
| 2 | BMI | 0.146717 |
| 3 | PhysicalActivity | 0.146175 |
| 4 | AlcoholIntake | 0.143387 |
| 5 | CancerHistory_0 | 0.139377 |
| 6 | GeneticRisk_2 | 0.110380 |
| 7 | Gender_0 | 0.064629 |
| 8 | Smoking_1 | 0.043402 |
| 9 | Smoking_0 | 0.034600 |
| 10 | Gender_1 | 0.011841 |
| 11 | GeneticRisk_0 | 0.006100 |
| 12 | GeneticRisk_1 | 0.000000 |
| 13 | CancerHistory_1 | 0.000000 |

Age was the highest-ranked feature according to the selected model's feature importance values.

*Note: Feature importance represents the model's internal splitting behavior. It does not establish medical causation or prove that a feature causes cancer.*

---

# 💾 10. Model Saving and Loading

The selected model was saved using Joblib.

### Save Model

```python
import joblib

joblib.dump(
    best_dt_model,
    'cancer_prediction_model.pkl'
)

print("Model saved successfully!")
```

### Load Model

```python
import joblib

loaded_model = joblib.load(
    'cancer_prediction_model.pkl'
)
```

### Test the Loaded Model

```python
loaded_predictions = loaded_model.predict(X_test)

print("Loaded Model Predictions:")
print(loaded_predictions[:10])
```

### Loaded Model Evaluation

```python
from sklearn.metrics import accuracy_score

loaded_accuracy = accuracy_score(
    y_test,
    loaded_predictions
)

print("Loaded Model Accuracy:", loaded_accuracy)
```

**Result:**

```text
Loaded Model Accuracy: 0.8933333333333333
```

The saved model was successfully loaded and produced the same test accuracy of 89.33%.

The saved pipeline includes both preprocessing and the trained Decision Tree classifier.

---

# 📝 11. Project Conclusion

In this project, a Machine Learning-based Cancer Diagnosis Prediction model was developed using a dataset containing 1,500 records and 9 columns.

The project involved data understanding, data quality analysis, Exploratory Data Analysis, statistical analysis, data preprocessing, and classification model development.

Logistic Regression and Decision Tree Classifier were implemented and evaluated. Hyperparameter tuning was performed using GridSearchCV with 5-fold cross-validation.

Based on the final test evaluation, the Tuned Decision Tree Classifier was selected as the preferred model.

It achieved:

- Accuracy: 89.33%
- Precision: 83.76%
- Recall: 88.29%
- F1-score: 85.96%
- ROC-AUC: 0.923

The selected model was successfully saved using Joblib and tested after loading.

Overall, this project demonstrates an end-to-end Machine Learning workflow for a binary classification problem, covering data analysis, statistical techniques, model development, optimization, evaluation, and model persistence.

---

# ⚠️ 12. Project Limitations

- The dataset contains only 1,500 records, which may limit the model's ability to generalize to larger and more diverse populations.
- The model has not been validated using an independent external dataset.
- The meanings of encoded categorical values and measurement units depend on the original dataset documentation.
- Model performance may vary when applied to new and unseen data.
- The model has not undergone clinical validation.

**This project is intended strictly for educational and research purposes. It must not be used for real medical diagnosis or clinical decision-making.**

---

# 🚀 13. Future Improvements

The following improvements can be considered for future development:

1. **Larger Dataset:** Use a larger and more diverse dataset to improve generalization ability.
2. **Additional Machine Learning Algorithms:** Explore Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Gradient Boosting.
3. **Feature Selection:** Apply feature selection techniques to identify relevant features.
4. **Advanced Hyperparameter Optimization:** Explore RandomizedSearchCV and Bayesian Optimization.
5. **External Validation:** Evaluate the selected model using an independent dataset.
6. **Model Deployment:** Develop a demonstration application using Flask or Streamlit.
7. **Model Explainability:** Apply SHAP or other explainability techniques to understand model predictions.
8. **Continuous Improvement:** Monitor model performance and update the model using reliable new data.

These improvements could enhance the model's reliability, interpretability, and practical usefulness in future research. Any future medical application would require appropriate clinical validation.

---

# 📁 14. Project Structure

The suggested GitHub repository structure is:

```text
Cancer-Diagnosis-Prediction/
│
├── data/
│   └── The_Cancer_data_1500_V2.csv
│
├── notebooks/
│   └── cancer_prediction.ipynb
│
├── models/
│   └── cancer_prediction_model.pkl
│
├── README.md
│
├── requirements.txt
│
└── .gitignore
```

*Note: Upload the dataset and trained model only if you have permission to share them. Do not publish sensitive or restricted data.*

---

# 📦 15. Installation and Requirements

### Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels joblib jupyter
```

### Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
joblib
jupyter
```

---

# ▶️ 16. How to Run the Project

1. Clone or download this GitHub repository.
2. Install the required Python libraries.
3. Open Jupyter Notebook.
4. Open `notebooks/cancer_prediction.ipynb`.
5. Ensure that the dataset path is correctly configured.
6. Run the notebook cells in sequence.
7. Load the saved model using Joblib for prediction.

---

# 👩‍💻 17. Author

**Machine Learning Project – Cancer Diagnosis Prediction**

Developed as an educational Machine Learning project to demonstrate practical skills in:

- Python Programming
- Data Cleaning
- Exploratory Data Analysis
- Statistical Analysis
- Data Preprocessing
- Machine Learning
- Hyperparameter Tuning
- Model Evaluation
- Feature Importance Analysis
- Model Saving and Loading

---

# ⚖️ Disclaimer

This project is intended strictly for educational and research purposes. It is not a clinically validated diagnostic system and should not be used to make medical decisions.

Any real-world medical application would require extensive clinical validation, independent testing, and appropriate regulatory review.
