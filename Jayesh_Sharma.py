#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd

data = pd.read_csv(r"C:\Users\win 10\Downloads\housing.csv", header=None, sep='\s+')
column_list = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data.columns = column_list
data.head()


# In[18]:


data.isnull().sum()


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[20]:


x_vars = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
y_vars = ['MEDV']
g = sns.PairGrid(data, x_vars=x_vars, y_vars=y_vars)
g.fig.set_size_inches(25, 3)
g.map(sns.scatterplot)
g.add_legend()


# In[21]:


plt.figure(figsize=(20, 10))
sns.heatmap(data.corr(),  annot=True)


# In[22]:


from sklearn.model_selection import train_test_split

boston = data[['INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'LSTAT', 'PTRATIO', 'MEDV']]

features = boston.drop('MEDV', axis=1)
labels = boston['MEDV']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=43)

X_train.shape, X_test.shape, y_train.shape


# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_linear = model.predict(X_test)

print('MAE:', mean_absolute_error(y_pred_linear, y_test))
print('MSE:', mean_squared_error(y_pred_linear, y_test))
print('R2_score:', r2_score(y_pred_linear, y_test))


# In[24]:


sns.regplot(x=y_pred_linear, y=y_test)
plt.xlabel('predict MEDV')
plt.ylabel('MEDV')


# In[25]:


from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(n_estimators=100, random_state=54, max_depth=10)
regr.fit(X_train, y_train)

y_pred_rnd = regr.predict(X_test)
print('MAE:', mean_absolute_error(y_pred_rnd, y_test))
print('MSE:', mean_squared_error(y_pred_rnd, y_test))
print('R2_score:', r2_score(y_pred_rnd, y_test))


# In[26]:


feat_importances = pd.DataFrame(regr.feature_importances_, index=X_train.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.plot(kind='bar', figsize=(8,6))


# In[27]:


sns.regplot(x=y_pred_rnd, y=y_test)
plt.xlabel('predict MEDV')
plt.ylabel('MEDV')
plt.xlim(5, 50)


# In[28]:


df = pd.DataFrame({'prediction': y_pred_rnd, 'test data': y_test, 'error': y_pred_rnd - y_test})
df.head()


# In[29]:


df[df['error'].abs() >= 5]


# In[ ]:




