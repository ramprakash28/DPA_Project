Churn Analysis with Machine Learning

Overview

This project focuses on analyzing customer churn using advanced machine learning techniques. The objective is to identify key factors contributing to churn and build predictive models to classify customers as churned or non-churned. Insights from this analysis aim to assist businesses in customer retention strategies.

Project Objectives

Explore relationships between features and their impact on churn.
Visualize patterns and separability using dimensionality reduction techniques like PCA, UMAP, and t-SNE.
Apply unsupervised learning for clustering and hidden pattern discovery.
Train supervised models to classify churn and optimize them through hyperparameter tuning.
Propose actionable insights to improve customer retention.

Dataset

Source: https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3/data 

Description: The dataset contains information about customers of a telecom company. It includes features such as demographics, subscription details, and usage patterns. The target variable Churn Label indicates whether a customer has churned (Yes) or not (No).

Size: 7,043 rows and 22 columns.

Use Case: Predict customer churn to enhance retention strategies.


Project Workflow

Exploratory Data Analysis (EDA):
Heatmaps, scatter plots, and dimensionality reduction visualizations.
Key correlations: Total Revenue ↔ Total Charges, Satisfaction Score ↔ Churn Score.
Feature Engineering:
Interaction terms: Combined Tenure and Monthly Charge.
Transformations: Log scaling of Monthly Charge.
Dimensionality Reduction:
Techniques: PCA, UMAP, and t-SNE.
Fine-tuned parameters to improve cluster separability.

Modeling

Algorithms: Random Forest, Logistic Regression, and XGBoost.
Cross-validation: Stratified K-Fold for robust evaluation.
Hyperparameter tuning using GridSearchCV.
Performance Evaluation:
Metrics: Accuracy, Precision, Recall, F1-Score.

Results

Model	Accuracy	Precision	Recall	F1-Score
Random Forest (Tuned)	0.9514	0.9631	0.8497	0.9027
Logistic Regression	0.9496	0.9616	0.8438	0.8988
XGBoost (Tuned)	0.9509	0.9536	0.8566	0.9024

Key Insights

Dimensionality reduction highlighted moderate separability between churned and non-churned customers.
Feature engineering significantly improved model performance.
Random Forest with hyperparameter tuning was the best-performing model.

Potential Pitfalls

Overfitting: Addressed using cross-validation and early stopping.
Class Imbalance: Mitigated with careful train-test splits.
Interpretability: Some models like XGBoost require additional tools (e.g., SHAP) for feature importance.

Future Work

Incorporate additional features to capture behavioral patterns.
Explore ensemble models like stacking or blending.
Deploy the model in production and monitor performance over time.

Dependencies

Python 3.7 or above
Required Libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
umap-learn
License

This project is licensed under the MIT License.

Acknowledgments

Ram Prakash Bollam - A20561314

Praveenraj Seenivasan - A20562374

Deekshitha Adishesha Raje Urs - A20560682

Nihal Korukanti - A20562326

Prithwee Reddei Patelu - A20560828

Data Source: Telco Customer Churn Dataset on Kaggle by Alfath Terry.

Special thanks to the Kaggle community for providing this valuable dataset for research and experimentation.
