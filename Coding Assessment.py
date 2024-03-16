#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[102]:


df=pd.read_csv('train.csv')
df1= pd.read_csv("test.csv")
df3=pd.read_csv('sample_submission.csv')


# In[37]:


from bs4 import BeautifulSoup  
# For HTML removal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[26]:


df1


# In[103]:


df3


# In[22]:


df.shape


# In[57]:


df1.shape


# In[13]:


df.info()


# In[20]:


df.tail()


# In[40]:


def remove_html_tags(text):
  soup = BeautifulSoup(text, "html.parser")
  return soup.get_text()  # Extract text without tags

df["New_Sentence"] = df["New_Sentence"].apply(remove_html_tags)
df1["New_Sentence"] = df1["New_Sentence"].apply(remove_html_tags)


# In[41]:


df


# In[75]:


df


# In[49]:


import string

def clean_text(text):
  text = text.lower()  # Convert to lowercase
  text = ''.join([c for c in text if c not in string.punctuation])  # Remove punctuation

  return text


# In[51]:


df["New_Sentence"] = df["New_Sentence"].apply(clean_text)
df1["New_Sentence"] = df1["New_Sentence"].apply(clean_text)


# In[69]:


df1


# # Feature Engineering

# In[65]:


vectorizer = TfidfVectorizer(max_features=1000)
features_df = vectorizer.fit_transform(df["New_Sentence"])


# In[66]:


features_df1 = vectorizer.transform(df1["New_Sentence"])


# In[68]:


features_df1 


# In[67]:


features_df


# # Logistic Regression

# In[73]:


model = LogisticRegression(multi_class="ovr", solver="lbfgs")
model.fit(features_df, df["Type"])


# In[109]:


X_train, X_val, y_train, y_val = train_test_split(features_df, df["Type"], test_size=0.2, random_state=80)


# In[110]:


predictions = model.predict(X_val)


# In[111]:


accuracy = accuracy_score(y_val, predictions)
print("Accuracy on validation set:", accuracy)


# In[112]:


predictions_test = model.predict(features_df)


# In[113]:


predictions_test


# In[93]:


from sklearn.model_selection import GridSearchCV


# In[101]:


param_grid = {'max_features': [2000]}


# In[114]:


grid_search = GridSearchCV(estimator=TfidfVectorizer(), param_grid=param_grid)


# In[115]:


from sklearn.neighbors import KNeighborsClassifier


# In[108]:


mode2 = KNeighborsClassifier(n_neighbors=500) 
mode2.fit(features_df, df["Type"])


# In[117]:


X_train, X_val, y_train, y_val = train_test_split(features_df,df["Type"], test_size=0.2, random_state=80)
predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print("Accuracy on validation set:", accuracy)


# In[ ]:




