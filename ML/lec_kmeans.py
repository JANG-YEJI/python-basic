#!/usr/bin/env python
# coding: utf-8

# # Kmeans
# * 1. 중심 포인트(CP1랜덤)로 부터 k(임의지정)개를 그룹화
# * 2. 그 그룹화된 반경 내에서 중심(CP1)과 각 포인트(k-1) 간의 거리(유클리드)를 최소화 하는 군집 형성
# * 3. 평균 거리값이 가장 작은 포인트가 중심으로 재설정(CP2)
# * 4. 중심(CP1)은 일반 포인트가 되어 가자 가까운 CP를 찾고 반경내로 재그룹이 형성
# * 5. 포인트의 변경이 없을 때 까지 무한 반복 

# In[1]:


from sklearn.cluster import KMeans
import matplotlib.pyplot  as plt
import seaborn as sns

from sklearn import datasets
import pandas as pd

from sklearn.metrics import silhouette_score, silhouette_samples


# In[2]:


def myscore(model, X, ncluster): 
    c_coef = silhouette_samples(X, ncluster)
    print(c_coef[:5])
    df["c_coef"] = c_coef
    
    score = silhouette_score(X, ncluster)
    print(score)
    
    print(f'응집도:  {model.inertia_:.4f}')


# In[3]:


iris = datasets.load_iris()
cols = columns=['Sepal length','Sepal width','Petal length','Petal width']


df = pd.DataFrame(iris.data, columns=cols)
df['target'] = iris.target

df.head()


# In[4]:


df_2col = df[['Sepal length','Sepal width']]
df_2col.head()

# create model and prediction
model = KMeans(n_clusters=3)
model.fit(df_2col)
ncluster = model.predict(df_2col)
df['ncluster'] = ncluster

print(df)


# In[5]:


model.cluster_centers_


# In[6]:


centers = pd.DataFrame(model.cluster_centers_, columns=['Sepal length','Sepal width'])
center_x = centers['Sepal length']
center_y = centers['Sepal width']

# scatter plot
plt.scatter(df['Sepal length'],df['Sepal width'],c=df['ncluster'],alpha=0.5)
plt.scatter(center_x,center_y,s=50,marker='D',c='r')
plt.show()


# In[ ]:





# In[7]:


c_coef = silhouette_samples(df[['Sepal length', 'Sepal width']], df["ncluster"])
print(c_coef[:5])
df["c_coef"] = c_coef


# In[8]:


df.head()


# In[9]:


print(f'응집도:  {model.inertia_:.4f}')


# In[ ]:





# In[10]:


myscore(model, df[['Sepal length', 'Sepal width']], df["ncluster"])


# In[ ]:





# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


scaler = StandardScaler()
scaler_val = scaler.fit_transform(df[['Sepal length', 'Sepal width']])

# df_scaler = pd.DataFrame(scaler_val, columns = ['Sepal length', 'Sepal width'])
# df_scaler.head()

df[['Sepal_length_scaler', 'Sepal_width_scaler']] = scaler_val


# In[13]:


df_2col = df[['Sepal_length_scaler', 'Sepal_width_scaler']]
#df_2col.head()

# create model and prediction
#model = KMeans(n_clusters=3)
model.fit(df_2col)
ncluster = model.predict(df_2col)
df['ncluster_scaler'] = ncluster


# In[14]:


df.head()


# In[15]:


myscore(model, df[['Sepal_length_scaler', 'Sepal_width_scaler']], df['ncluster_scaler'])


# In[16]:


centers = pd.DataFrame(model.cluster_centers_, columns=['Sepal_length_scaler', 'Sepal_width_scaler'])
center_x = centers['Sepal_length_scaler']
center_y = centers['Sepal_width_scaler']

# scatter plot
plt.scatter(df['Sepal_length_scaler'],df['Sepal_width_scaler'],c=df['ncluster_scaler'],alpha=0.5)
plt.scatter(center_x,center_y,s=50,marker='D',c='r')
plt.show()


# In[ ]:





# In[ ]:





# # 최적의 군집 개수는 몇개?

# In[24]:


NS = [2, 3, 4, 5, 6]
df_2col = df[['Sepal_length_scaler', 'Sepal_width_scaler']]
for i in NS:
    model = KMeans(n_clusters=i)
    model.fit(df_2col)
    ncluster = model.predict(df_2col)
    #df['ncluster_scaler'] = ncluster
    print(i, "--"*15)
    myscore(model, df_2col, ncluster)
    
    centers = pd.DataFrame(model.cluster_centers_, columns=['Sepal_length_scaler', 'Sepal_width_scaler'])
    center_x = centers['Sepal_length_scaler']
    center_y = centers['Sepal_width_scaler']
    #print(df['ncluster_scaler'].unique())

    # scatter plot
    plt.scatter(df['Sepal_length_scaler'],df['Sepal_width_scaler'],c=df['ncluster_scaler'],alpha=0.5)
    plt.scatter(center_x,center_y,s=50,marker='D',c='r')
    plt.show()


# In[ ]:





# # 검증

# In[19]:


ctab = pd.crosstab(df["target"], df["ncluster_scaler"])
print(ctab)


# In[22]:


df.sort_values('ncluster_scaler', ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:





# # 계층적 군집

# In[25]:


from scipy.cluster.hierarchy import linkage, dendrogram


# In[27]:


plt.figure(figsize=(20,20))
#y, method='single', metric='euclidean',
matrix = linkage(df[['Sepal_length_scaler', 'Sepal_width_scaler']], method='single', metric='euclidean')
dendrogram(matrix)
plt.show()


# In[ ]:




