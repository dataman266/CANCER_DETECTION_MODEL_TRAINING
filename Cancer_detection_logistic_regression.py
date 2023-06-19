#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


# In[60]:


cancerds = load_breast_cancer()


# In[61]:


cancerds


# In[62]:


cancerds.keys()


# In[63]:


cancerds.data


# In[64]:


cancerds.target


# In[65]:


cancerds.feature_names


# In[66]:


df = pd.DataFrame(data = cancerds.data, columns = cancerds.feature_names)
df


# In[67]:


df['Target'] = cancerds.target


# In[68]:


df


# In[69]:


df.head()


# In[70]:


df.tail(2)


# In[71]:


df.sample()


# In[72]:


df.dtypes


# In[73]:


df


# In[74]:


# x = df.drop(['target'], axis = 1)
x = df.iloc[:,0:-1]
x


# In[75]:


y = df.iloc[:,-1]
y


# In[76]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)


# In[77]:


x_train.shape


# In[78]:


y_train.shape


# In[79]:


x_test.shape


# In[80]:


y_test.shape


# In[81]:


lg = LogisticRegression()


# In[82]:


lg.fit(x_train, y_train)


# In[83]:


pred = lg.predict(x_test)
pred


# In[84]:


accuracy_score(y_test, pred)


# In[85]:


print(confusion_matrix(y_test, pred))


# In[86]:


print(classification_report(y_test, pred))


# In[87]:


sns.countplot(df['Target'])


# In[88]:


plt.figure(figsize = (20,10))
sns.heatmap(df.corr(), annot = True)


# In[89]:


p = confusion_matrix(y_test, pred)
sns.heatmap(p, annot = True)


# In[90]:


df.to_csv('cancer.csv')


# In[91]:


def cancerpredict(p):
    p = p.reshape(1, -1)
    pred = lg.predict(p)
    print("Predicted value: ", pred)
    if pred == 0:
        print("No Cancer")
    else:
        print("Cancer Detected")


# In[93]:


df


# In[ ]:





# In[95]:


p = np.array([7.76,24.54,47.92,181.0,0.05263,0.04362,0.0,0.0,0.1587,0.05884,0.3857,1.428,2.548,19.15,0.007189,0.00466,0.0,0.0,0.02676,0.002783,9.456,30.37,59.16,268.6,0.08996,0.06444,0.0,0.0,0.2871,0.07039])
cancerpredict(p)


# In[96]:


p = np.array([20.6,29.33,140.1,1265.0,0.1178,0.277,0.3514,0.152,0.2397,0.07016,0.726,1.595,5.772,86.22,0.006522,0.06158,0.07117,0.01664,0.02324,0.006185,25.74,39.42,184.6,1821.0,0.165,0.8681,0.9387,0.265,0.4087,0.124])
cancerpredict(p)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




