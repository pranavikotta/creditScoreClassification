import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#obtain data
all_credit_data = pd.read_csv('Credit_Score_Clean.csv.zip', index_col=0)

#identify features relevant to analysis and filter original dataset accordingly
fico_features = {'payment_history': ['Delay_from_due_date', 'Num_of_Delayed_Payment', 'Payment_Behaviour', 'Payment_of_Min_Amount'], 
                 'amounts_owed': ['Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Num_of_Loan'],
                 'length_of_credit_history': ['Credit_History_Age_Months'], 'new_credit':['Num_Credit_Inquiries'],
                 'credit_mix': ['Credit_Mix']}

selected_features = sum(fico_features.values(), []) #flattens into a single list of lists, omitting keys
credit_data = all_credit_data.loc[:, selected_features]
print(credit_data.head())
#print(credit_data.dtypes)

#label encoding for categorical data
le = LabelEncoder()
credit_score_order = ['Poor', 'Standard', 'Good']
le.fit(credit_score_order)
all_credit_data['Credit_Score'] = le.transform(all_credit_data['Credit_Score'])

credit_mix_order = ['Bad', 'Standard', 'Good']
le.fit(credit_mix_order)
credit_data['Credit_Mix'] = le.transform(credit_data['Credit_Mix'])

credit_data['Payment_Behaviour'] = le.fit_transform(credit_data['Payment_Behaviour'])
credit_data['Payment_of_Min_Amount'] = le.fit_transform(credit_data['Payment_of_Min_Amount'])

#data preprocessing - normalization of numerical features
numerical_cat = credit_data.select_dtypes(include=['int64', 'float64']).columns
credit_data[numerical_cat] = StandardScaler().fit_transform(credit_data[numerical_cat])

#set up test-train split
X = credit_data
y = all_credit_data['Credit_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#train model: random forest classifier, using 200 trees
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)
#check accuracy on test dataset
print(f'RFC Model Accuracy: {rf.score(X_test, y_test)}')

#train model: eXtreme Gradient Boosting
xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=15)
xgb.fit(X_train, y_train)
print(f'XGB Model Accuracy: {xgb.score(X_test, y_test)}')

#train model: support vector classifier
svc = SVC()
svc.fit(X_train, y_train)
print(f'SVC Model Accuracy: {svc.score(X_test, y_test)}')

#ensemble learning
voting_clf = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='hard')
voting_clf.fit(X_train, y_train)
print(f'Voting Classifier Accuracy: {voting_clf.score(X_test, y_test)}')