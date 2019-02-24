
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
print(os.listdir(r"C:\Users\Harshubh Meherishi\Desktop\Folders\KaggleInput"))


# In[3]:


df=pd.read_csv(r"C:\Users\Harshubh Meherishi\Desktop\Folders\KaggleInput\train.csv\train.csv", engine='python')


# In[4]:


df.groupby('Category').size()


# In[5]:


sns.countplot(df['Category'],label="Count")
plt.show()


# In[6]:


df.isna().sum()


# In[7]:


X=df.drop(['Category'],axis=1)  #Seperating the label as y and the rest of attributes in X
y=df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)


# In[8]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[9]:


df_test=pd.read_csv(r"C:\Users\Harshubh Meherishi\Desktop\Folders\KaggleInput\test.csv")
Predictions=logreg.predict(df_test)
df_test['Category']=Predictions
Final_Dataframe=df_test[['Id','Category']]
Final_Dataframe.to_csv('Submission_1.csv',index=False)

