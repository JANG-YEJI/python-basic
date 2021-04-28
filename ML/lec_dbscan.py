#!/usr/bin/env python
# coding: utf-8

# # DBSCAN (Density Based)
# * 밀집도 기반의 군집 형성
# * k-Means와 달라 k없이 군집 가능 
# * leaf부터 시작해서 최종 1개의 그룹을 형성할 때 까지 클러스터링
# 
# 
# * 그룹화 조건 : eps: 3, n:5
# * 모든 클러스터링은 원의 반경(3)이 동일하다.
#     * [1]Center 노드: 3개의 노드 수
#     * [2]Border 노드: 2개의 노드수 (1개:포인트 노드, 1개: Center노드) --> 드랍X
#     * [3]Noise 노드: 2개의 노드 수 (1개: 포인트 노드, 1개: 포인트 노드) --> 그룹핑에서 배제
#     
#     * [1] Center 노드: 5개의 노드 수 (3개의 노드 수 + [2])

# In[13]:


from sklearn.cluster import DBSCAN
import matplotlib.pyplot  as plt
import seaborn as sns

from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples


# In[2]:


iris = datasets.load_iris()
cols = columns=['Sepal length','Sepal width','Petal length','Petal width']


df = pd.DataFrame(iris.data, columns=cols)
df['target'] = iris.target

df.head()


# In[47]:


X_feature = df[['Petal length', 'Petal width']]


# In[98]:


dbscan = DBSCAN(eps = 0.3, min_samples=7, metric='euclidean')
pred_cluster = dbscan.fit_predict(X_feature)

df["pred_cluster"] = pred_cluster
df["labels_"] = dbscan.labels_


# In[99]:


df.head()


# In[100]:


y_label_ = df["labels_"]
c_coef = silhouette_samples(X_feature, y_label_)
print("실루엣 계수(각 node별): ", c_coef[:5])
avg_coef = np.mean(c_coef)
print("실루엣 계수(전체 노드) 평균: ", avg_coef)


# In[101]:


avg_coef = silhouette_score(X_feature, y_label_)
print("실루엣 계수(전체 노드) 평균: ", avg_coef)


# In[ ]:





# In[ ]:





# # 검증: 교차 테이블 

# In[102]:


ctab = pd.crosstab(df["target"], df["pred_cluster"])
print(ctab)
#-1이 noise 


# In[ ]:





# In[ ]:





# In[113]:


eps = [0.1, 0.2, 0.3, 0.5, 1, 1.2, 1.5]
for e in eps:
    dbscan = DBSCAN(eps = e, min_samples=15, metric='euclidean')
    pred_cluster = dbscan.fit_predict(X_feature)

    ctab = pd.crosstab(df["target"], pred_cluster)
    print(e, "--" * 20, "\n", ctab)
    
    plt.scatter(df['Petal length'], df['Petal width'], c=pred_cluster)
    plt.title(silhouette_score(X_feature, dbscan.labels_)) #score가 낮을 수록 좋음 
    # coef = np.mean(센터노드 - 각 점들간의 거리)
    # silhouette: mean(센터노드 - 각 점들간의 거리) / max(거리) == 1 (정규분포)
    # 실루엣은 1에 가까울수록 좋다
    
    # -1: 노이즈 노드(반경에 속하지 못하는 노드)
    plt.show()


# In[ ]:




