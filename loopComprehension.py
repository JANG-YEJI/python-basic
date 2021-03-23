# 기본 for문
numbers = [1, 2, 3, 4, 5]
square = []
for i in numbers:
    square.append(i**2)
print(square)

# 리스트 컴프리헨션 방법
square = [i**2 for i in numbers]
print(square)



# 조건문을 포함하는 for문
square = []
for i in numbers:
    if i >= 3:
        square.append(i**2)
print(square)

# 리스트 컴프리헨션 방법
square = [i**2 for i in numbers if i >= 3]
print(square)