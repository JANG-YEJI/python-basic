#!/usr/bin/env python
# coding: utf-8

# 1. 뉴스 속보 수집
#     - 이미지 링크
#     - 기사 링크, 기사 제목
#     - 기사 내용
#     
#     
# 2. 뉴스 속보 데이터 만들기(DataFrame)
#     - 이미지링크, 기사링크, 기사제목, 기사내용
#     - 변수1, 변수2, 변수3, 변수4
#     - 리스트로 만들고 append
#     
#     
# 3. 이미지 링크에서 이미지 데이터 다운받기
#     - img_down 폴더 만들고
#     - 이미지 저장하기

# In[25]:


import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[23]:


headers = { "User-Agent":"Mozilla/5.0"}
url = "http://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001"
html = requests.get(url, headers = headers).text
soup = BeautifulSoup(html, "html.parser")
all = soup.find("ul", {"class" : "type06_headline"})
dl = all.find_all("dl")

list_img, list_link, list_title, list_contents = ([] for i in range(4))

for item2 in dl:
    try: #이미지 링크
        img = item2.find("dt", {"class" : "photo"}).find("img")
        list_img.append(img["src"])
        print(img["src"])
    except:
        print("No image") 
        
    #기사 링크, 기사 제목    
    link = item2.find("dt", {"class" : ""}).find("a")
    list_link.append(link["href"])
    print(link["href"])
    list_title.append(link.text.replace("\t", "").replace("\n", "")[1:len(link.text)+1])
    print(link.text.replace("\t", "").replace("\n", "")[1:len(link.text)+1])
    
    try: #기사 내용
        content = item2.find("dd")
        list_contents.append(content.text.replace("\t","").replace("\n","").split("...")[0])
        print(content.text.replace("\t","").replace("\n","").split("...")[0])
    except:
        print("No Content")
    
    print ("-------------------------------")


# In[16]:


list_img


# In[21]:


list_title


# In[22]:


list_link


# In[24]:


list_contents


# In[26]:


df = pd.DataFrame({
    '이미지 링크': list_img,
    '기사 링크': list_link,
    '기사 제목': list_title,
    '기사 내용': list_contents
})
df.head()


# In[ ]:





# In[28]:


# https://data-make.tistory.com/170 [Data Makes Our Future] 
# 폴더 만들기 
import os
 
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


# In[29]:

folder_path = ""
createFolder(folder_path)


# In[ ]:





# In[42]:


# https://soyoung-new-challenge.tistory.com/92
# 이미지 저장 
import urllib.request
import re

for i in range(len(list_img)):
    url = list_img[i]
    
    # url에서 특정 문자열 추출
    regex = re.compile('{}(.*){}'.format(re.escape('origin/'), re.escape('.jpg')))
    text = regex.findall(url)
    text[0] = text[0].replace("/", "_")
    
    #저장
    urllib.request.urlretrieve(url, folder_path + "/{}.jpg".format(text[0]))

