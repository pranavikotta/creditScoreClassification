# creditScoreClassification
Leveraging ML to classify credit scores, drawing on the FICO model.

Overview:

This project aims to predict an individual's credit score based on various credit-related features. The dataset used in this project was sourced from Kaggle, and the task is to build a machine learning model that classifies individuals into credit score categories: Poor, Standard, and Good. A set of machine learning classifiers, including Random Forest, XGBoost, and Support Vector Classifier (SVC), are employed to build the model, with an ensemble learning approach used to combine some of the models for improved performance.

Features:

The features used in the model are selected based on the FICO (Fair Isaac Corporation) scoring model, which evaluates creditworthiness based on several factors: Payment History, Amounts Owed, Length of Credit History, New Credit, Credit Mix.

Selected Features:

From the original Kaggle dataset, relevant features were selected and grouped according to the FICO model. The dataset was filtered to include these features, ensuring relevance to the task.
- Payment History: Delay_from_due_date, Num_of_Delayed_Payment, Payment_Behaviour, Payment_of_Min_Amount
- Amounts Owed: Outstanding_Debt, Credit_Utilization_Ratio, Total_EMI_per_month, Num_of_Loan
- Length of Credit History: Credit_History_Age_Months
- New Credit: Num_Credit_Inquiries
- Credit Mix: Credit_Mix

Preprocessing:

1. Label Encoding:
Categorical features such as Payment_Behaviour and Payment_of_Min_Amount were transformed into numerical values using Label Encoding, allowing the machine learning models to process these features effectively.
2. Feature Normalization:
Numerical features were standardized using StandardScaler. This ensures that all numerical features are on the same scale, which is particularly important for models like Support Vector Machines (SVM).
3. Train-Test Split:
The dataset was split into training (80%) and testing (20%) sets to evaluate the models' performance on unseen data.

Observations and Next Steps:

The models achieve an accuracy of around 70-75%, which is a reasonable start for predicting credit scores. However, the model performance could be improved further. Hyperparameter tuning using GridSearchCV could help in improving the model accuracy by finding the optimal parameters for each algorithm. Model Stacking: Combining multiple models in a more sophisticated manner (e.g., stacking) could potentially increase accuracy by leveraging their complementary strengths. Further feature engineering could improve model performance. More granular features or external data (e.g., credit utilization trends over time) could potentially be integrated into the model.
