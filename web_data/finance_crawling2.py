#!/usr/bin/env python
# coding: utf-8

# In[1]:
import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[28]:
headers = { "User-Agent":"Mozilla/5.0"}
code = "005930"
url = "https://finance.naver.com/item/main.nhn?code=005930"

html = requests.get(url, headers = headers)
soup = BeautifulSoup(html.text)
sub_section = soup.find_all("tbody")

tbody = sub_section[2]
td = tbody.find_all("td")


# In[30]:


table_list = []
for item in td:
    table_value = item.text.replace("\n", "").replace("\t", "")
    table_list.append(table_value)


# In[31]:


n = 10
result_list = []
for row in range((len(table_list) + n -1) // n):
    data = table_list[row * n:(row + 1) * n]
    result_list.append(data)


# In[72]:


df = pd.DataFrame(result_list)
df


# In[64]:


col = soup.select("#content > div.section.cop_analysis > div.sub_section > table > thead > tr")
col_th = col[1].find_all("th")


# In[67]:


col_list = []
for item in col_th:
    table_value = item.text.replace("\n", "").replace("\t", "")
    col_list.append(table_value)


# In[66]:


col_th[0].text


# In[68]:


col_list


# In[73]:


df.columns = col_list


# In[75]:


th = tbody.find_all("th")


# In[82]:


row_list = []
for item in th:
    table_value = item.text
    row_list.append(table_value)


# In[83]:


row_list


# In[84]:


df.index = row_list


# In[85]:

df
