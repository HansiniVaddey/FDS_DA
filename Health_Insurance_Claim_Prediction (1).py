#!/usr/bin/env python
# coding: utf-8

# # Health Insurance Claim Prediction
# 
# 
# Claim Prediction: The model can predict whether a policyholder is likely to make a claim based on their demographic information, medical history, and lifestyle factors. Insurance companies can use these predictions to anticipate future claim volumes and allocate appropriate provisions accordingly.
# 
# 
# This model serves as a valuable tool for insurance companies to optimize claim provisions, assess risk exposure, detect fraud, make informed underwriting decisions, and manage capital effectively, ultimately contributing to the financial stability and sustainability of the company.
# 

# ### Data Preprocessing
# 
# Steps:
# 
# 1.Import the necessary libraries
# 
# 
# 2.Import the dataset
# 
# 
# 3.Handling null values
# 
# 
# 4.Data Visualization
# 
# 
# 5.Outlier detection
# 
# 
# 6.Seperate Dependent and independent variables
# 
# 
# 7.Encoding
# 
# 
# 8.Feature scaling
# 
# 
# 9.Splitting into training and testing set
# 
# 

# ###1.Import the necessary libraries
# 

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle

# ###2.Import the dataset
# 

# In[ ]:


dataset=pd.read_csv("Health_Insurance15.csv")


# In[ ]:


dataset


# In[ ]:


dataset.head()


# In[ ]:


dataset.tail()


# In[ ]:


dataset.shape


# In[ ]:


dataset.info()


# In[ ]:


dataset.describe()


# In[ ]:


corr=dataset.corr()
corr


# In[ ]:


plt.subplots(figsize=(20,15))
sns.heatmap(corr,annot=True)


# In[ ]:


dataset.sex.value_counts()


# In[ ]:


dataset.age.value_counts()


# In[ ]:


dataset.diabetes.value_counts()


# In[ ]:


dataset.smoker.value_counts()


# In[ ]:





# ## 3.Handling null values
# 
# 

# In[ ]:


dataset.isnull().any()


# In[ ]:


dataset["age"].fillna(dataset["age"].mean(),inplace=True)


# In[ ]:


dataset["bmi"].fillna(dataset["bmi"].mode()[0],inplace=True)


# In[ ]:


dataset


# In[ ]:


dataset.isnull().any() #Handled all the null values


# In[ ]:





# ## 4.Data Visualization
# It involves creating graphical representations of data to help individuals, including data scientists and stakeholders, better understand patterns, trends, and insights within the dataset.

# ### i) Scatter plot

# In[ ]:


sns.scatterplot(x="diabetes",y="weight",data=dataset)


# In[ ]:


sns.scatterplot(x="regular_ex",y="weight",data=dataset)


# In[ ]:


sns.scatterplot(x="smoker",y="age",data=dataset)


# In[ ]:





# ### ii)Line Plot

# In[ ]:


sns.lineplot(x="claim",y="age",data=dataset)


# In[ ]:


sns.lineplot(x="sex",y="claim",data=dataset)


# In[ ]:


sns.lineplot(x="smoker",y="claim",data=dataset)


# ### iii)Distribution plot

# In[ ]:


sns.displot(dataset["bmi"])


# In[ ]:


sns.displot(dataset["city"])


# In[ ]:


sns.displot(dataset["age"])


# In[ ]:





# ### iii)Relational Plot
# 

# In[ ]:


sns.relplot(x="claim",y="bmi",data=dataset,hue='sex')


# In[ ]:





# ### iv)Bar Plot
# 

# In[ ]:


sns.barplot(data=dataset, x="smoker", y="claim", width=0.5)


# In[ ]:





# In[ ]:





# In[ ]:


sns.barplot(data=dataset, x="sex", y="bloodpressure", width=0.5)


# In[ ]:





# ### v)Joint Plot

# In[ ]:


sns.jointplot(x="sex",y="bmi",data=dataset)


# ### vi)Box Plot
# 

# In[ ]:


sns.boxplot(x="age",y="sex",data=dataset)


# # 5.Outlier detection
# Outliers are those data points that are significantly different from the rest of the dataset. They are often abnormal observations that skew the data distribution, and arise due to inconsistent data entry, or erroneous observations.

# In[ ]:


sns.boxplot(dataset.age)


# In[ ]:


sns.boxplot(dataset.bmi)


# In[ ]:


#outliers present in bmi


# In[ ]:


q1=dataset.bmi.quantile(0.25)
q3=dataset.bmi.quantile(0.75)


# In[ ]:


print(q1)
print(q3)


# In[ ]:


IQR=q3-q1
IQR


# In[ ]:


upper_limit=q3+1.5*IQR
upper_limit


# In[ ]:


dataset.bmi.median()


# In[ ]:


dataset['bmi']= np.where(dataset['bmi']>upper_limit,14.4542,dataset['bmi'])


# In[ ]:


dataset['bmi']


# In[ ]:


sns.boxplot(dataset.bmi)


# In[ ]:


#MOST OF THE OUTLIERS REDUCED


# ## 6.Seperate dependent and independent variables

# In[ ]:


#Independent Variables
x=dataset.drop(columns=['claim'], inplace=False)


# In[ ]:


x


# In[ ]:


x.head()


# In[ ]:


#Dependent variable

y=dataset.iloc[:,12:]


# In[ ]:


y


# In[ ]:


dataset.shape


# In[ ]:


x.shape


# In[ ]:


y.shape


# ## 7.Encoding
# Convert the strings into numerical or binary format

# **Label Encoding on city**

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


l=LabelEncoder()


# In[ ]:


x["city"]=l.fit_transform(x["city"])
x["city"]


# In[ ]:


x.head()


# In[ ]:


#Similarly changing sex,hereditary_diseases,job_title


# In[ ]:


x["sex"]=l.fit_transform(x["sex"])
x["sex"]


# In[ ]:


x["hereditary_diseases"]=l.fit_transform(x["hereditary_diseases"])
x["hereditary_diseases"]


# In[ ]:


x["job_title"]=l.fit_transform(x["job_title"])
x["job_title"]


# In[ ]:


x.head()


# In[ ]:





# ##### Correlation after encoding

# In[ ]:


sns.heatmap(corr,annot=True)


# ## 8.Feature Scaling
# 
# Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. It is performed during the data pre-processing to handle highly varying magnitudes or values or units. If feature scaling is not done, then a machine learning algorithm tends to weigh greater values, higher and consider smaller values as the lower values, regardless of the unit of the values.
# 
# Its only done on independent varaibles because with change in independent variables the magnitude will change.Which will indirectly change dependent variables.
# 
# 2 Types:
# 
# Standard scaling => mean=0 && sd=1
# 
# min max scaling  => range from 0 to 1
# 

# In[ ]:


#feature scaling
from sklearn.preprocessing import MinMaxScaler
ms=MinMaxScaler()
x_scaled=pd.DataFrame(ms.fit_transform(x),columns=x.columns)


# In[ ]:


x_scaled


# ## 9.Splitting into training set and testing set
# 
# We have training and testing data
# 
# example: 1000 rows
# 
# Training data : 70%-80%
# 
# Testing data  : 20%-30% (checking performance)

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[ ]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[ ]:


x_train.head()


# In[ ]:





# ### Model Building
# 1.Import the model building Libraries
# 
# 2.Initializing the model
# 
# 3.Training and testing the model
# 
# 4.Evaluation of Model
# 
# 5.Save the Model

# 
# **Logistic regression** can be applied to the Health_Insurance dataset to predict whether a claim will be made or not based on various features such as age, weight, BMI, hereditary diseases, number of dependents, smoker status, city, blood pressure, diabetes, regular exercise, and job title.
# 
# **Binary logistic regression** is used to predict the probability of a binary outcome, such as yes or no, true or false, or 0 or 1.
# 
# **Why logistic Regression**
# We use logistic regression for your Health_Insurance dataset because we are dealing with a binary classification problem. Logistic regression is a statistical method used for modeling the relationship between a binary dependent variable (the outcome or response variable, typically represented as 0 or 1) and one or more independent variables (predictors or features). It estimates the probability that the dependent variable is a particular category based on the values of the independent variables, such as predicting whether a claim will be made or not based on various features.
# 

# 
# **Steps in Logistic Regression**
# 
# Model Training: Fit a logistic regression model to the training data. This involves estimating the coefficients (parameters) of the logistic regression equation using the training data.
# 
# 
# Model Evaluation: Evaluate the performance of the logistic regression model on the testing data using appropriate evaluation metrics such as accuracy, precision, recall, F1-score, or ROC AUC.
# 
# 
# Interpretation: Interpret the coefficients of the logistic regression model to understand the relationship between the features and the likelihood of making a claim.

# # Logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:


model = LogisticRegression()  # Example model, you can use any other classifier
model.fit(x_train, y_train)

# Predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)




pickle.dump(model,open("Health_Insurance_Claim_Prediction(1).pkl","wb"))