#!/usr/bin/env python
# coding: utf-8

#  - https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA


# ## 데이터 확인

# In[3]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sub_df = pd.read_csv('sample_submission.csv')


# In[4]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[5]:


print(train.shape, test.shape)


# In[ ]:





# ### 결측치 확인

# In[8]:


nan_dict = {"CNT":train.isna().sum(),
            "RATE": train.isna().sum()/train.shape[0] *100
}
nan_df = pd.DataFrame(nan_dict)
print(nan_df[nan_df["RATE"]>0].sort_values("CNT", ascending=False))

# PoolQC        1453  99.520548     Pool quality                                              NA	No Pool
# MiscFeature   1406  96.301370     Miscellaneous feature not covered in other categories,    NA	None
# Alley         1369  93.767123     Type of alley access to property,                         NA 	No alley access
# Fence         1179  80.753425     Fence quality,                                            NA	No Fence
# FireplaceQu    690  47.260274     Fireplace quality                                         NA	No Fireplace
# LotFrontage    259  17.739726     Linear feet of street connected to property

# GarageType      81   5.547945     Garage location                          NA	No Garage         
# GarageYrBlt     81   5.547945     Year garage was built
# GarageFinish    81   5.547945     Interior finish of the garage            NA	No Garage
# GarageQual      81   5.547945     Garage quality                           NA	No Garage
# GarageCond      81   5.547945     Garage condition                         NA	No Garage     

# BsmtExposure    38   2.602740      Refers to walkout or garden level walls                 NA	No Basement     No	No Exposure
# BsmtFinType2    38   2.602740      Rating of basement finished area (if multiple types)    NA	No Basement
# BsmtFinType1    37   2.534247      Rating of basement finished area                        NA	No Basement
# BsmtCond        37   2.534247      Evaluates the general condition of the basement         NA	No Basement
# BsmtQual        37   2.534247      Evaluates the height of the basement                    NA	No Basement


# In[9]:


nan_dict = {"CNT":test.isna().sum(),
            "RATE": test.isna().sum()/test.shape[0] *100
}
nan_df = pd.DataFrame(nan_dict)
print(nan_df[nan_df["RATE"]>0].sort_values("CNT", ascending=False))

# train에는 없는 null 
# MSZoning         4   0.274160            Identifies the general zoning classification of the sale.
# BsmtFullBath     2   0.137080            Basement full bathrooms
# BsmtHalfBath     2   0.137080
# Functional       2   0.137080
# Utilities        2   0.137080
# GarageCars       1   0.068540
# GarageArea       1   0.068540
# TotalBsmtSF      1   0.068540
# KitchenQual      1   0.068540
# BsmtUnfSF        1   0.068540
# BsmtFinSF2       1   0.068540
# BsmtFinSF1       1   0.068540
# Exterior2nd      1   0.068540
# Exterior1st      1   0.068540
# SaleType         1   0.068540


# #### 불필요한 피쳐 삭제

# In[10]:


nan_features_list = nan_df[nan_df["RATE"]>50]
print(nan_features_list.index)


# In[11]:


train.drop(nan_features_list.index, axis=1, inplace=True)
test.drop(nan_features_list.index, axis=1, inplace=True)


# In[12]:


train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)


# In[13]:


print(train.shape, test.shape)


# In[ ]:





# ### 상관분석

# In[14]:


corr_s = train.corrwith(train["SalePrice"]).sort_values(ascending=False)
print(corr_s)


# #### 불필요한 피쳐 삭제

# In[15]:


corr_features_list = corr_s[abs(corr_s)<0.2]
print(corr_features_list.index)


# In[16]:


train.drop(corr_features_list.index, axis=1, inplace=True)
test.drop(corr_features_list.index, axis=1, inplace=True)


# In[17]:


print(train.shape, test.shape)


# In[ ]:





# ### 결측치 채우기

# In[18]:


nan_dict = {"CNT":train.isna().sum(),
            "RATE": train.isna().sum()/train.shape[0] *100,
            "TYPE": train.dtypes
}
nan_df = pd.DataFrame(nan_dict)
print(nan_df[nan_df["RATE"]>0].sort_values("CNT", ascending=False))

# null 의미 
# LotFrontage   259  17.739726  float64 --> mean 
# GarageType     81  5.547945   object  NO
# GarageYrBlt    81  5.547945  float64  Year garage was built --> 0             corr    GarageYrBlt      0.486362
# GarageFinish   81  5.547945   object  NO
# GarageQual     81  5.547945   object  NO
# GarageCond     81  5.547945   object  NO
# BsmtExposure   38  2.602740   object  NO
# BsmtFinType2   38  2.602740   object  NO
# BsmtQual       37  2.534247   object  NO
# BsmtCond       37  2.534247   object  NO
# BsmtFinType1   37  2.534247   object  NO
# MasVnrType      8  0.547945   object  Masonry veneer area in square feet  --> None 
# MasVnrArea      8  0.547945  float64  Masonry veneer type  --> 0 
# Electrical      1  0.068493   object  Electrical system  ---> 최빈값


# In[19]:


train[["GarageYrBlt", "MasVnrArea"]] = train[["GarageYrBlt", "MasVnrArea"]].fillna(0)
test[["GarageYrBlt", "MasVnrArea"]] = test[["GarageYrBlt", "MasVnrArea"]].fillna(0)


# In[20]:


train["LotFrontage"].fillna(train["LotFrontage"].mean(), inplace=True)
test["LotFrontage"].fillna(test["LotFrontage"].mean(), inplace=True)


# In[21]:


train["MasVnrType"].fillna("None", inplace=True)
test["MasVnrType"].fillna("None", inplace=True)


# In[22]:


nan_dict = {"CNT":train.isna().sum(),
            "RATE": train.isna().sum()/train.shape[0] *100,
            "TYPE": train.dtypes
}
nan_df = pd.DataFrame(nan_dict)
print(nan_df[nan_df["RATE"]>0].sort_values("CNT", ascending=False))


# In[23]:


train[nan_df[nan_df["RATE"]>0].index] = train[nan_df[nan_df["RATE"]>0].index].fillna("NO")
test[nan_df[nan_df["RATE"]>0].index] = test[nan_df[nan_df["RATE"]>0].index].fillna("NO")


# In[24]:


nan_dict = {"CNT":train.isna().sum(),
            "RATE": train.isna().sum()/train.shape[0] *100,
            "TYPE": train.dtypes
}
nan_df = pd.DataFrame(nan_dict)
print(nan_df[nan_df["RATE"]>0].sort_values("CNT", ascending=False))


# In[25]:


nan_dict = {"CNT":test.isna().sum(),
            "RATE": test.isna().sum()/test.shape[0] *100,
            "TYPE": test.dtypes
}
nan_df = pd.DataFrame(nan_dict)
print(nan_df[nan_df["RATE"]>0].sort_values("CNT", ascending=False))


# In[26]:


nan_index = nan_df[nan_df["RATE"]>0].index


# In[27]:


test[nan_index] = test[nan_index].fillna(test.mode().iloc[0])


# In[29]:


nan_dict = {"CNT":test.isna().sum(),
            "RATE": test.isna().sum()/test.shape[0] *100,
            "TYPE": test.dtypes
}
nan_df = pd.DataFrame(nan_dict)
print(nan_df[nan_df["RATE"]>0].sort_values("CNT", ascending=False))


# In[30]:


train.info()


# In[31]:


test.info()


# In[32]:


cols = train.select_dtypes(include='float').columns
print(cols)
train[cols] = train[cols].astype(int)


# In[33]:


cols = test.select_dtypes(include='float').columns
print(cols)
test[cols] = test[cols].astype(int)


# In[ ]:





# In[ ]:





# In[ ]:





# ### Object --> Numeric

# In[34]:


train.select_dtypes(include='object')


# In[35]:


object_index = train.select_dtypes(include='object').columns   # 38개


# In[36]:


for col in object_index:
    print(col, train[col].unique())


# In[37]:


print(train.shape, test.shape)


# In[ ]:





# In[38]:


le_encoder = LabelEncoder()

cols = object_index
for col in cols:   
    le_encoder.fit(train[col])
    train["LE_"+ col] = le_encoder.transform(train[col])
    test["LE_"+ col] = le_encoder.transform(test[col])

train.head()


# In[39]:


test.head()


# In[40]:


train.drop(object_index, axis=1, inplace=True)
test.drop(object_index, axis=1, inplace=True)


# In[41]:


print(train.shape, test.shape)


# In[42]:


train.info()


# In[43]:


test.info()


# #### 불필요한 피쳐 삭제

# In[44]:


corr_s = train.corrwith(train["SalePrice"]).sort_values(ascending=False)
print(corr_s[abs(corr_s)<0.2])


# In[45]:


corr_features_list = corr_s[abs(corr_s)<0.2]

train.drop(corr_features_list.index, axis=1, inplace=True)
test.drop(corr_features_list.index, axis=1, inplace=True)
print(train.shape, test.shape)


# In[46]:


corr_s = train.corrwith(train["SalePrice"]).sort_values(ascending=False)
print(corr_s)


# In[ ]:





# ### Scaler

# In[47]:


train.columns


# In[48]:


train.iloc[:, 0:23].describe()


# In[49]:


# fig, axes = plt.subplots(ncols=8, nrows=3, figsize=(30,20))
# for i, feature in enumerate(train.columns):
#     cols = i%8
#     rows = i//8     
#     sns.distplot(train[feature], kde=True, ax=axes[rows][cols])
#     axes[rows][cols].set_title(feature)


# In[50]:


scale_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd',
       'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
       '2ndFlrSF', 'GrLivArea', 'GarageYrBlt',  'GarageArea',
       'WoodDeckSF', 'OpenPorchSF', 'SalePrice']
for sf in scale_features:
    train[sf] = np.log1p(train[sf])
train.head()    


# In[51]:


scale_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd',
       'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
       '2ndFlrSF', 'GrLivArea', 'GarageYrBlt',  'GarageArea',
       'WoodDeckSF', 'OpenPorchSF']
for sf in scale_features:
    test[sf] = np.log1p(test[sf])
test.head()    


# In[52]:


corr_s = train.corrwith(train["SalePrice"]).sort_values(ascending=False)
print(corr_s)


# In[53]:


corr_features_list = corr_s[abs(corr_s)<0.2]
print(corr_features_list.index)


# In[54]:


train.drop(corr_features_list.index, axis=1, inplace=True)
test.drop(corr_features_list.index, axis=1, inplace=True)


# In[55]:


print(train.shape, test.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[56]:


print(train.columns)
print(test.columns)


# In[57]:


train.info()


# In[58]:


test.info()


# In[59]:


# cols = train.select_dtypes(include='float').columns
# print(cols)
# train[cols] = train[cols].astype(int)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 점수확인

# ### columns 37

# In[88]:


X_df = train.drop(["SalePrice"], axis=1)
y_df = train["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.2, shuffle=True, random_state = 121)
print(X_df.shape, y_df.shape)
print(X_train.shape, y_train.shape)


# In[90]:


# LinearRegression()
lasso = Lasso(alpha=1.0)    # a*|W|
ridge = Ridge(alpha=1.0)    # a*W^2
elsnet = ElasticNet(alpha=1.0, l1_ratio=0.2)  # Lasso + Ridge      # 0.2(a*|W|) + 0.8(a*W^2)


# In[91]:


models = [lasso, ridge, elsnet]
for model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    mse = mean_squared_error(y_val, pred, squared=True)
    rmse = mean_squared_error(y_val, pred, squared=False)
    print(f'{model.__class__.__name__} MSE:{mse:.5f} RMSE:{rmse:.5f}') 
    
# Lasso MSE:0.15270 RMSE:0.39077
# Ridge MSE:0.01522 RMSE:0.12338
# ElasticNet MSE:0.07162 RMSE:0.26761


# In[92]:


cv_model = GridSearchCV(ridge, param_grid={'alpha':[0.01, 0.05, 1.0, 5.0, 10]},
            scoring='neg_root_mean_squared_error',
            cv = 10, refit=True)
cv_model.fit(X_train, y_train)
print(f'RMSE: {cv_model.best_score_ * -1:.5f}')
print(cv_model.best_params_)


# In[93]:


pred = cv_model.predict(X_val)  
#print(pred)
mse = mean_squared_error(y_val, pred, squared=True)
rmse = mean_squared_error(y_val, pred, squared=False)
print(f'MSE:{mse:.5f} RMSE:{rmse:.5f}')

# MSE:0.01502 RMSE:0.12257


# In[ ]:





# In[ ]:





# In[94]:


pred = cv_model.predict(test)


# In[100]:


print(pred) #sclaer된 pred

predo = np.round(np.expm1(pred), 6)
print(predo)


# In[101]:


sub_df


# In[102]:


print(sub_df.shape)
print(len(predo))


# In[103]:


sub_df["SalePrice"] = np.array(predo).reshape(-1, 1)


# In[104]:


sub_df


# In[105]:


sub_df.to_csv("gcv_submission.csv", index=False)


# In[ ]:





# ### columns 28

# In[106]:


corr_s = train.corrwith(train["SalePrice"]).sort_values(ascending=False)
print(corr_s)


# In[107]:


corr_features_list = corr_s[abs(corr_s)<0.3]
print(corr_features_list.index)


# In[108]:


train.drop(corr_features_list.index, axis=1, inplace=True)
test.drop(corr_features_list.index, axis=1, inplace=True)

print(train.shape, test.shape)


# In[ ]:





# In[109]:


X_df = train.drop(["SalePrice"], axis=1)
y_df = train["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.2, shuffle=True, random_state = 121)
print(X_df.shape, y_df.shape)
print(X_train.shape, y_train.shape)


# In[110]:


cv_model = GridSearchCV(ridge, param_grid={'alpha':[0.01, 0.05, 1.0, 5.0, 10]},
            scoring='neg_root_mean_squared_error',
            cv = 10, refit=True)
cv_model.fit(X_train, y_train)
print(f'RMSE: {cv_model.best_score_ * -1:.5f}')
print(cv_model.best_params_)

pred = cv_model.predict(X_val)  
#print(pred)
mse = mean_squared_error(y_val, pred, squared=True)
rmse = mean_squared_error(y_val, pred, squared=False)
print(f'MSE:{mse:.5f} RMSE:{rmse:.5f}')

# RMSE: 0.15108
# {'alpha': 0.01}
# MSE:0.01786 RMSE:0.13365


# In[ ]:





# In[ ]:





# In[ ]:





# ## PCA 

# In[72]:


pca = PCA(n_components=20)
#pipeline = make_pipeline(std_scaler, pca)
pca_res = pca.fit_transform(X_df)
print(pca_res.shape)
print("주성분(PC) 15개가 전체 데이터를 얼마나 설명할수 있는가?\n", np.sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_)


# In[73]:


pca_df = pd.DataFrame(data=pca_res)
pca_df["SalePrice"] = y_df
pca_df
# 어떠한 피쳐가 합쳐진건지는 알 수 없음 


# In[74]:


pca_df_y = pca_df["SalePrice"]  #df["traget"]  
pca_df_X = pca_df[pca_df.columns.difference(["SalePrice"])]
X_train_pca, X_val_pca, y_train_pca, y_val_pca = train_test_split(pca_df_X, pca_df_y, test_size=0.2,  random_state=36, shuffle=True)

models = [lasso, ridge, elsnet]
for model in models:
    model.fit(X_train_pca, y_train_pca)
    pred = model.predict(X_val_pca)
    mse = mean_squared_error(y_val_pca, pred, squared=True)
    rmse = mean_squared_error(y_val_pca, pred, squared=False)
    print(f'{model.__class__.__name__} MSE:{mse:.5f} RMSE:{rmse:.5f}') 

# n = 20
# Lasso MSE:0.18575 RMSE:0.43099
# Ridge MSE:0.02982 RMSE:0.17268
# ElasticNet MSE:0.07380 RMSE:0.27166

# n = 15    
# Lasso MSE:0.18575 RMSE:0.43099
# Ridge MSE:0.03514 RMSE:0.18745
# ElasticNet MSE:0.07380 RMSE:0.27166

# n = 10
# Lasso MSE:0.18575 RMSE:0.43099
# Ridge MSE:0.03728 RMSE:0.19307
# ElasticNet MSE:0.07380 RMSE:0.27166


# In[ ]:




