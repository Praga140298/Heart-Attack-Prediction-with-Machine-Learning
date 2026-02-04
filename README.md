# Heart-Attack-Prediction-with-Machine-Learning
End‑to‑End Machine Learning pipeline for Heart Attack Prediction

This project builds an end‑to‑end machine learning pipeline to predict the presence of heart disease using a clinical dataset of 294 patients. The work covers data cleaning, exploratory data analysis (EDA), feature engineering, and benchmarking several classification models with different preprocessing strategies.

### Dataset

`Source`: data.csv (heart attack prediction dataset)

`Rows`: 294 patients

`Columns`: 14 features

### Features

**Numerical**

- `age`: Age in years  
- `trestbps`: Resting blood pressure  
- `chol`: Serum cholesterol  
- `thalach`: Maximum heart rate achieved  
- `oldpeak`: ST depression induced by exercise  

**Categorical / discrete**

- `sex`: Biological sex (0 = female, 1 = male)  
- `cp`: Chest pain type (1–4)  
- `fbs`: Fasting blood sugar  
- `restecg`: Resting electrocardiographic results  
- `exang`: Exercise-induced angina  


**Target:**
  

    num - Heart disease presence (0 = no disease, 1 = disease)
    

The raw dataset contains missing values encoded as "?" in several columns, especially chol, fbs, restecg, thalach, exang, slope, ca, and thal.

**Data Cleaning**

The raw heart attack dataset included 14 features with mixed data types and missing values stored as `"?"` instead of proper nulls. To simplify modeling, columns with extremely high missingness (`slope`, `ca`, `thal`) were removed. For the remaining features, `"?"` values were converted to actual missing values and clinical measurements such as resting blood pressure (`trestbps`), cholesterol (`chol`), maximum heart rate (`thalach`), and ST depression (`oldpeak`) were cast to numeric. Missing values were then imputed using sensible statistics: mean for continuous vitals (e.g., `trestbps`, `thalach`), median for skewed measurements (`chol`), and mode for binary/categorical flags (`fbs`, `restecg`, `exang`). The target column name was also cleaned to remove trailing spaces so that `num` consistently represents heart disease status. After these steps, the dataset had no remaining missing values in the retained features and was ready for EDA and model training.

    
**Data Quality Report**

The dataset contains 294 observations and 14 variables, with missing values originally encoded as "?"

Columns with extremely high missingness (slope, ca, thal) were dropped to avoid unreliable imputation

Remaining missing values were handled with type-aware strategies: mean for trestbps and thalach, median for chol, and mode for fbs, restecg, and exang

After cleaning, all retained features are complete (no missing values), correctly typed, and show reasonable ranges and variability suitable for modeling


### Exploratory Data Analysis (EDA)

To understand the cleaned dataset, I first standardized the main numerical features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) using z‑scores and visualized their distributions with histograms and KDE curves. Most variables showed roughly unimodal shapes, with some right‑skewness and clear extreme values, especially in cholesterol, resting blood pressure, and ST depression. Boxplots on the standardized scale were then used to highlight potential outliers and quantify how many extreme points each feature contained.

Next, I examined relationships between features and the target (`num`) using a correlation matrix and a sorted bar plot of feature–target correlations. Chest pain type (`cp`), exercise‑induced angina (`exang`), ST depression (`oldpeak`), maximum heart rate (`thalach`), and sex (`sex`) emerged as the most informative predictors of heart disease, while some features showed weaker direct correlation. The correlation heatmap also revealed moderate multicollinearity between several predictors (for example, between `exang` and `oldpeak` or `cp` and `exang`), suggesting that regularized linear models or tree‑based algorithms would be appropriate. Finally, scatter‑matrix plots and boxplots stratified by the target class confirmed clear distributional differences between patients with and without heart disease, supporting the suitability of this dataset for supervised classification.

### Model Setup

The target variable num was modeled as a binary classification problem (presence vs. absence of heart disease).
Features were split into train and test sets with stratification to preserve class balance.
Multiple models were benchmarked: Logistic Regression (L2), KNN, SVM with RBF kernel, Decision Tree, Random Forest, Gradient Boosting, and Gaussian Naive Bayes.
Each model was embedded in two preprocessing pipelines: one with StandardScaler and one with RobustScaler to assess the impact of scaling and outlier robustness.


| Model                          | Scaling Strategy |
| ------------------------------ | ---------------- |
| Logistic Regression (L2)       | StandardScaler   |
| K-Nearest Neighbors (KNN, k=7) | StandardScaler   |
| SVM (RBF kernel)               | RobustScaler     |
| Decision Tree                  | StandardScaler   |
| Random Forest (300 trees)      | RobustScaler     |
| Gradient Boosting              | StandardScaler   |
| Gaussian Naive Bayes           | RobustScaler     |


### Evalution

Each pipeline was evaluated on the test set using accuracy, precision, recall, F1-score, and ROC–AUC, leveraging class probabilities or decision functions where available.
Results were aggregated into a comparison table and sorted primarily by ROC–AUC, then F1, then accuracy to prioritize ranking by discrimination and balance between precision and recall.
SVM with RBF kernel (with StandardScaler) and KNN emerged among the strongest performers, achieving high ROC–AUC (around 0.90+) and competitive accuracy and F1 scores; Random Forest and Naive Bayes also performed well.
Overall, the evaluation shows that properly scaled kernel-based and ensemble models can provide accurate heart disease prediction on this cleaned clinical dataset, while robust scaling offers an alternative when outliers are a concern.

### Key Takeaways

Thoughtful data cleaning and imputation (mean/median/mode) are crucial when working with real clinical data containing missing and noisy values.

EDA reveals that chest pain type, exercise-induced angina, ST depression, maximum heart rate, and sex are informative predictors of heart disease.

Multicollinearity among predictors motivates the use of regularized linear models or tree-based/ensemble methods.

Benchmarking multiple models with different scaling strategies shows that SVM (RBF), KNN, and Random Forest provide strong predictive performance on this dataset.

### Future Work

Hyperparameter tuning (GridSearchCV/RandomizedSearchCV) for top models

Cross-validation for more robust performance estimates

Feature selection or dimensionality reduction (e.g., RFE, PCA)

Calibration of predicted probabilities and threshold optimization

Deployment as an API or simple web app for clinical decision support (educational/demo purpose only)





