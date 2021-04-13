import pandas as pd

# Series(): 라벨을 갖는 1차원 데이터 
s1 = pd.Series([10, 20, 30, 40, 50])
print("s1.index: ", s1.index)
print("s1.values: ", s1.values)
print(s1)

# 인덱스를 따로 지정해서 Series 만들기
alpha_index = ['a', 'b', 'c']
s2 = pd.Series([1, 2, 3], index =alpha_index)
print(s2)

# dic형태로 Series 만들기
s3 = pd.Series({'국어':100, '영어':95, '수학':90})
print(s3)

# 날짜 자동 생성 date_range
print(pd.date_range(start='2021-01-01', end='2021-01-07'))
print(pd.date_range(start = '2020-01-01 10:00', periods = 4, freq='30min'))

# DataFrame 생성
df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(df1)

# 인덱스를 따로 지정해서 DataFrame 만들기
list1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index_date = pd.date_range('2021-01-01', periods=3)
columns_list = ['A', 'B', 'C']
df2 = pd.DataFrame(list1, index = index_date, columns=columns_list)
print(df2)

# dic형태로 DataFrame 만들기
table_data = {'연도': [2015, 2016, 2016, 2017],
            '지사': ['한국', '한국', '미국', '한국'],
            '고객 수': [200, 250, 450, 300]}
df3 = pd.DataFrame(table_data)
print(df3)
print(pd.DataFrame(table_data, columns = ['지사', '연도', '고객 수']))

# Series 연산
s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([10, 20, 30, 40, 50, 60])
print(s1+s2)

# DataFrame 연산
table_data1 = {'A': [1, 2, 3, 4, 5],
               'B': [10, 20, 30, 40, 50],
               'C': [100, 200, 300, 400, 500],
               'D': [1000, 2000, 3000, 4000, 5000]}
df1 = pd.DataFrame(table_data1)

table_data2 = {'A': [6, 7, 8],
               'B': [60, 70, 80],
               'C': [600, 700, 800]}
df2 = pd.DataFrame(table_data2)
print(df1+df2)

print(df2.mean()) #평균
print(df2.mean(axis=1))
print(df2.describe())



# index로 원하는 값 불러오기
list1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index_date = pd.date_range('2021-01-01', periods=3)
columns_list = ['A', 'B', 'C']
df3 = pd.DataFrame(list1, index = index_date, columns=columns_list)
#print(df3.loc['A'])   #-------- column이름에는 사용할 수 없다.
print(df3.loc['2021-01-01'])
print(df3['A']['2021-01-01'])