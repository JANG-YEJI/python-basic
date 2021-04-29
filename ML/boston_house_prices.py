#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, fbeta_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold


# In[3]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[4]:


from sklearn.datasets import load_boston


# In[5]:


# X, y = load_boston(return_X_y=True)
# print(X, y)

dataset = load_boston()
print(dataset.keys())
df = pd.DataFrame(data=dataset.data,
    columns=dataset.feature_names
)
df["MEDV"] = dataset.target #kaggle target MEDV
X_df = df.iloc[: , :-1]
y_df = df.iloc[: , -1]

X_train , X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.2, shuffle=True, random_state=121)
print(X_train.shape , y_train.shape,X_val.shape,  y_val.shape )


# In[6]:


# dataset = load_boston()
# df = pd.DataFrame(data = dataset.data,
#                  columns = dataset.feature_names)
# df["target"] = dataset.target
# X_df = df.iloc[:, :-1]
# y_df = df.iloc[:, -1]

# X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.2, shuffle=True, random_state = 121)
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)


# In[7]:


df.info()


# In[8]:


df.head()


# In[9]:


df.describe()


# -  이산형 (카테고리성 ) 수치는 스케일링 시키지 않는다. 

# <pre>
# [01]  CRIM 자치시(town) 별 1인당 범죄율  
# [02]  ZN 25,000 평방피트를 초과하는 거주지역의 비율  
# [03]  INDUS 비소매상업지역이 점유하고 있는 토지의 비율  
# [04]  CHAS 찰스강에 대한 더미변수(강의 경계에 위치한 경우는 1, 아니면 0)  
# [05]  NOX 10ppm 당 농축 일산화질소  
# [06]  RM 주택 1가구당 평균 방의 개수  
# [07]  AGE 1940년 이전에 건축된 소유주택의 비율  
# [08]  DIS 5개의 보스턴 직업센터까지의 접근성 지수  
# [09]  RAD 방사형 도로까지의 접근성 지수  
# [10]  TAX 10,000 달러 당 재산세율  
# [11]  PTRATIO 자치시(town)별 학생/교사 비율  
# [12]  B 1000(Bk-0.63)^2, 여기서 Bk는 자치시별 흑인의 비율을 말함.  
# [13]  LSTAT 모집단의 하위계층의 비율(%)  
# [14]  MEDV 본인 소유의 주택가격(중앙값) (단위: $1,000)  --> target
# </pre>

# In[10]:


df.hist()
plt.show()


# In[ ]:





# In[ ]:





# <pre>
# * skelarn.metrics.mean_squared_error
# squared = True   MSE
# squared = False  RMSE
# </pre>

# In[11]:


model1 = LinearRegression()
model1.fit(X_train, y_train)
pred = model1.predict(X_val)
mse = mean_squared_error(y_val, pred, squared=True)
rmse = mean_squared_error(y_val, pred, squared=False)
print(f'MSE:{mse:.5f} RMSE:{rmse:.5f}') 
# MSE:26.27265 RMSE:5.12569
# 5000달러 오차


# In[ ]:





# ## preprocessing (전처리)

# In[12]:


plt.figure(figsize=(9, 5))
sns.heatmap(df.corr(), annot=True, fmt=".2g")


# In[13]:


df.corrwith(df["MEDV"]).sort_values(ascending=False)
# RM         0.695360
# ZN         0.360445
# B          0.333461

# NOX       -0.427321
# TAX       -0.468536
# INDUS     -0.483725
# PTRATIO   -0.507787
# LSTAT     -0.737663


# In[14]:


nfeature = ['RM','ZN','B','NOX','TAX','INDUS','PTRATIO','LSTAT']


# In[15]:


sns.pairplot(df[nfeature])


# In[17]:


fig, axes = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
for i, feature in enumerate(nfeature):
    cols = i%7
    rows = i//7     
    sns.boxplot(x="MEDV", y=feature, data=df, ax=axes[rows][cols])
    axes[rows][cols].set_title(feature)


# In[ ]:





# In[18]:


fig, axes = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
for i, feature in enumerate(df.columns):
    cols = i%7
    rows = i//7     
    sns.distplot(df[feature], kde=True, ax=axes[rows][cols])
    axes[rows][cols].set_title(feature)


# In[19]:


# df.columns
scale_features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT', 'MEDV']
for sf in scale_features:
    df[sf] = np.log1p(df[sf])


# In[20]:


df.head()


# In[21]:


X_df = df.iloc[:, :-1]
y_df = df.iloc[:, -1]

X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.2, shuffle=True, random_state = 121)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)


# In[22]:


model1 = LinearRegression()
model1.fit(X_train, y_train)
pred = model1.predict(X_val)
mse = mean_squared_error(y_val, pred, squared=True)
rmse = mean_squared_error(y_val, pred, squared=False)
print(f'MSE:{mse:.5f} RMSE:{rmse:.5f}') 
# MSE:0.03310 RMSE:0.18193


# In[ ]:





# In[23]:


fig, axes = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
for i, feature in enumerate(df.columns):
    cols = i%7
    rows = i//7     
    sns.regplot(y=df['MEDV'], x=df[feature], ax=axes[rows][cols])
    axes[rows][cols].set_title(feature)


# In[24]:


# 회귀 계수
model1.coef_


# In[25]:


MEDV_corr= df.corrwith(df["MEDV"])
coef_df = pd.DataFrame(model1.coef_, 
                       index= X_df.columns, columns = ["coef"])
coef_df["corr"] = MEDV_corr
coef_df.sort_values(by="coef", ascending= False)
# 회귀값(coef)이 클수록 집값에 영향을 많이 준다. 
# NOX가 가장 영향을 많이 끼친다. 


# <pre>
# 회귀에서 규제란 언제 하나: 회귀계수가 너무 크면 잘맞춘다. --> 오버피팅
# 규제를 적용해서 0.5(NOX) 회귀값을 낮출 것이다. 
# 규제 l1은 절댓값을 사용하고
# 규제 l2는 제곱을 사용한다.
# </pre>

# <pre>
# Lasso : L1규제 a|W|, 피쳐 수 줄이기 
#     target에 중요도가 덜한 피쳐들은 규제 적용하면 w=0, 대상에서 제외
#     규제를 안해도 된다 --> 예측에서 안정적으로 사용되어질 피쳐
# Ridge : L2규제 aW^2, 피쳐수 그대로, 덜중요한 피쳐의 w가 작아진다
#     target에 중요도가 덜한 피쳐들은 규제 적용하면 w들은 0에 근사
# </pre>

# In[42]:


from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet


# In[27]:


# LinearRegression()
lasso = Lasso(alpha=1.0)    # a*|W|
ridge = Ridge(alpha=1.0)    # a*W^2
elsnet = ElasticNet(alpha=1.0, l1_ratio=0.2)  # Lasso + Ridge      # 0.2(a*|W|) + 0.8(a*W^2)


# In[28]:


models = [lasso, ridge, elsnet]
for model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    mse = mean_squared_error(y_val, pred, squared=True)
    rmse = mean_squared_error(y_val, pred, squared=False)
    print(f'{model.__class__.__name__} MSE:{mse:.5f} RMSE:{rmse:.5f}') 
    
# Lasso MSE:0.13108 RMSE:0.36205
# Ridge MSE:0.03366 RMSE:0.18348
# ElasticNet MSE:0.11899 RMSE:0.34495


# In[ ]:





# In[29]:


score_list = cross_val_score(model, X_df, y=y_df,
               scoring='neg_mean_squared_error',
               cv = 5)
score_list = -1 * score_list
print(score_list, score_list.mean())
# 0.16480502873757988


# In[30]:


from sklearn.model_selection import cross_validate


# In[31]:


for model in models:
    score_list = cross_validate(model, X_df, y=y_df,
                  scoring={'mse':'neg_mean_squared_error',
                          'rmse':'neg_root_mean_squared_error'},
                  cv=5, return_train_score=False)
    # print(score_list)
    mse_score = score_list['test_mse'] * -1
    rmse_score = score_list['test_rmse'] * -1
    print(f'MSE: {mse_score.mean():.5f}, RMSE:{rmse_score.mean():.5f}')


# In[ ]:





# In[32]:


cv_model = GridSearchCV(ridge, param_grid={'alpha':[0.01, 1.0, 5.0, 10]},
            scoring='neg_root_mean_squared_error',
            cv = 5, refit=True)
cv_model.fit(X_train, y_train)
print(f'RMSE: {cv_model.best_score_ * -1:.5f}')
print(cv_model.best_params_)


# In[52]:


pred = cv_model.predict(X_val)
print(pred) #sclaer된 pred

predo = np.round(np.expm1(pred), 1)
print(predo)


# In[ ]:





# In[ ]:





# In[33]:


alpha = [0.01, 1.0, 5.0, 10]
coef_df = pd.DataFrame(index = X_train.columns)
for a in alpha:
    ridge = Ridge(alpha = a)
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_val)
    score = mean_squared_error(y_val, pred, squared=False)
    
    coef_df["alpha"+ str(a)] = ridge.coef_
    print(f'alpha:{a} RMSE:{score:.5f}')
    
coef_df


# In[37]:


sns.countplot(data=coef_df, x= coef_df.columns)


# In[39]:


alpha = [0.01, 0.5, 1.0, 5.0, 10]
coef_df = pd.DataFrame(index = X_train.columns)
for a in alpha:
    for model in models:
        model = model(alpha = a)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = mean_squared_error(y_val, pred, squared=False)

        coef_df["alpha"+ str(a)] = model.coef_
        print(f'alpha:{a} RMSE:{score:.5f}')
    
coef_df


# In[43]:


alpha = [0.01, 1.0, 5.0, 10]
coef_df = pd.DataFrame(index = X_train.columns)
for a in alpha:
    lasso = Lasso(alpha = a)
    lasso.fit(X_train, y_train)
    pred = lasso.predict(X_val)
    score = mean_squared_error(y_val, pred, squared=False)
    
    coef_df["alpha"+ str(a)] = lasso.coef_
    print(f'alpha:{a} RMSE:{score:.5f}')
    
coef_df
# topn = ['ZN', 'INDUS', 'NOX', 'RM', 'RAD', 'TAX', 'PTRATIO']


# In[47]:





# In[ ]:




