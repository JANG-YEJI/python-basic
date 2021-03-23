print("< Division >")
a = 15
b = 6

# 나누기
c = a / b
print(str(a) + "/" + str(b) + "=", c)

# 몫
c = a // b
print(str(a) + "//" + str(b) + "=", c)

# 나머지
c = a % b
print(str(a)+ "%" + str(b) + "=", c)

# 제곱
c = 2**3
print("2**3 =", c)
print("-"*20)

#진수
print("< Base N >")
a = 18
print(str(a) + "의 2진수", bin(a))
print(str(a) + "의 8진수", oct(a))
print(str(a) + "의 16진수", hex(a))
print("-"*20)