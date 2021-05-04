#Objective: to predict whether a customer will churn or not
#Target Variable : Churn
#Problem- Classification - logistic regression

#data cleaning and preparation : merge, handle missing, one hot encoding, EDA, etc.
#feature selection
#model building - split data, model creation
#model_evalution - confusion matrix, accuracy score,precision and recall

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

churn = pd.read_csv('/Users/meghagupta/Desktop/IIBM/WA_Fn-UseC_-Telco-Customer-Churn.csv')

churn.head()
churn.info()
churn.describe()

#for finding which all columns have unique values  
for v in churn.columns:
    print()
    print(v, '---', churn[v].unique())
    
#taken all the columns which have 2 values that is yes and no
vars = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn']    

#converting yes/no to 1/0
def bin_mapping(x):
    return x.map({'Yes':1,'No':0})

churn[vars] = churn[vars].apply(bin_mapping)

churn.head()

#for converting other columns which has values instead of yes/no in numeric form:
for var in ['gender','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']:
    m1 = pd.get_dummies(churn[var],prefix=var)     #get_dummies : Convert categorical variable into dummy/indicator variables.
    churn = pd.concat([churn,m1],axis=1)
    
churn.drop(['gender','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod'],axis=1,inplace=True)
    
churn.columns

churn.drop(['MultipleLines_No phone service','OnlineSecurity_No internet service','OnlineBackup_No internet service','DeviceProtection_No internet service','TechSupport_No internet service','StreamingTV_No internet service','StreamingMovies_No internet service'],axis =1, inplace=True)

churn.info()

#customer id will not play any role so we can drop that also
churn.drop(['customerID'],axis=1,inplace=True)

churn.TotalCharges  #it data type is not numeric so need to convert into numeric form
churn.TotalCharges  = pd.to_numeric(churn.TotalCharges,errors='coerce')   #to_numeric: Convert argument to a numeric type and errors='coerce' will update missing value as NaN

churn.TotalCharges

churn.info()

#now dealing with null values of Total charges
churn.isnull().sum()

#we can drop the missing value NaN only if very low percentage of data has missing value
(churn.isnull().sum()/len(churn))*100   #to calculate how much % data is missing

#so only 0.156183% data is missing which is very low so we can drop that
churn= churn[-np.isnan(churn.TotalCharges)]

churn.info()

#split
X = churn.drop(['Churn'],axis = 1)
y = churn['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#Now Standard Scaling because some columns has 2 digit , some 3 , 4 , no uniform.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train [['tenure','MonthlyCharges','TotalCharges']]=sc.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])

import seaborn as sns
plt.figure(figsize=[18,10])     # for changing size of graph
sns.heatmap(churn.corr(), annot=True)   #heatmap is used to plot correlation

import statsmodels.api as sm
X1 = sm.add_constant(X_train)   
Logm1 = sm.GLS(y_train,X1)      #GLS is a logistic function/model
Logm1.fit().summary()

#RFE - Recursive feature Elimination for elinimating which feature which we can't use.
from sklearn.feature_selection import RFE
logreg = LogisticRegression()
rfe = RFE(logreg,15)         #we are taking 15 features
rfe.fit(X_train,y_train)

list(zip(X_train.columns,rfe.support_,rfe.ranking_))

col = X_train.columns[rfe.support_]   #columns which are true which is supporting

#now creating model now with support columns means on which output values depends
X2 = sm.add_constant(X_train[col])   
Logm2 = sm.GLM(y_train,X2,family=sm.families.Binomial())      #GLS is a logistic function/model
Logm2.fit().summary()

#the system will predict the value on given input
y_train_pred = Logm2.fit().predict(X2)
y_train_pred[:10]    #if value is greater than 0.5 then churn and if less than 0.5 then 

#comparing the given output and predicted output
y_train_pred_final = pd.DataFrame({'Churn': y_train.values, 'Chrun_Prob':y_train_pred})
y_train_pred_final ['CustID'] = y_train.index
y_train_pred_final.head()

y_train_pred_final['predicted'] = y_train_pred_final.Chrun_Prob.map(lambda x: 1 if x>0.5 else 0)

#confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_train_pred_final.Churn,y_train_pred_final.predicted)

accuracy_score(y_train_pred_final.Churn,y_train_pred_final.predicted)
