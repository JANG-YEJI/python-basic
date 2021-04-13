import numpy as np

data1 = [1, 2, 3, 4, 5]
a1 = np.array(data1)
print(data1)
print(a1) # array형은 ,가 없음

print(np.arange(0, 10, 2))
print(np.linspace(1, 10, 4)) # 1부터 10까지 4개의 수

data2 = [0, 1, 2, 3, 4]
a2 = np.array(data2)
print("list의 연산: ", data1 + data2)
print("array의 연산", a1 + a2)
print(a1 > a2)

print("list 요소의 합: ", sum(data1))
print("array 요소의 합: ", a1.sum())

print("누적 합 cumsum():", a1.cumsum())
print("누적 곱 cumprod(): ", a1.cumprod())

data_i_value = [data1[x] for x in [0, 1, 3]]
print("list 여러 요소 접근: ", data_i_value)
print("array 여러 요소 접근: ", a1[[0, 1, 3]])