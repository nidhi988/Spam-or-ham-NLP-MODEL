#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle


# In[ ]:


df=pd.read_csv("spam.csv")


# In[ ]:


#dropping unecessary columns
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)


# In[ ]:


df['label']=df['class'].map({'ham':0,'spam':1})


# In[ ]:


#dependent and independent variable
X=df['message']
y=df['label']


# In[ ]:


cv=CountVectorizer()
X=cv.fit_transform(X)


# In[ ]:


pickle.dump(cv,open('transform.pkl','wb'))


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


clf=MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
filename='nlp_model.pkl'
pickle.dump(clf,open(filename,'wb'))


# In[ ]:




