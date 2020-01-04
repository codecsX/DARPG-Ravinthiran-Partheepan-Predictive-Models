#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[4]:


train=pd.read_csv('C:/Users/RAVINTHIRAN/Documents/Dept_stat_receipt_disposal_010112019.csv')
test=pd.read_csv('C:/Users/RAVINTHIRAN/Documents/datafile.csv')
train['Type']='Train'
test['Type']='Test'
fullData = pd.concat([train,test],axis=0) 


# In[5]:


fullData.columns 
fullData.head(10) 
fullData.describe()


# In[17]:


ID_col = ['REF_NO']
target_col = ["Disposals.Month"]
cat_cols = ['Disposals','Month','Pending Between 2 To 6 Months','Pending Between 6 To 12 Months','Pending Less Than 2 Months','Pending More Than 1 Year','Recetpts']
num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(ID_col)-set(target_col))
other_col=['Type'] 


# In[9]:


fullData.isnull().any()


# In[10]:


num_cat_cols = num_cols+cat_cols


# In[11]:


for var in num_cat_cols:
    if fullData[var].isnull().any()==True:
        fullData[var+'_NA']=fullData[var].isnull()*1 


# In[12]:


fullData[num_cols] = fullData[num_cols].fillna(fullData[num_cols].mean(),inplace=True)


# In[19]:


for var in cat_cols:
 number = LabelEncoder()
 fullData[var] = number.fit_transform(fullData[var].astype('str'))


fullData["Disposals"] = number.fit_transform(fullData["Disposals"].astype('str'))

train=fullData[fullData['Type']=='Train']
test=fullData[fullData['Type']=='Test']

train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
Train, Validate = train[train['is_train']==True], train[train['is_train']==False]


# In[20]:


features=list(set(list(fullData.columns))-set(ID_col)-set(target_col)-set(other_col))


# In[23]:


x_train = Train[list(features)].values
y_train = Train["Month"].values
x_validate = Validate[list(features)].values
y_validate = Validate["Month"].values
x_test=test[list(features)].values


# In[25]:


random.seed(100)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)


# In[28]:


status = rf.predict_proba(x_validate)
fpr, tpr, _ = roc_curve(y_validate, status[:,1])
data = auc(fpr, tpr)
print (data)

final_status = rf.predict_proba(x_test)
test["Account.Status"]=final_status[:,1]
test.to_csv('C:/Users/Analytics Vidhya/Desktop/model_output.csv',columns=['REF_NO','Account.Status'])


# In[ ]:




