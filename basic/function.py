# 함수의 정의와 호출
def my_student_info(name, school_ID, phoneNumber):
    print("*"*20)
    print("* 학생이름: ", name)
    print("* 학급번호: ", school_ID)
    print("* 전화번호: ", phoneNumber)

my_student_info("현아", "01", "01-235-6789")
my_student_info("진수", "02", "01-987-6543")


# 람다 함수
print('\nlambda function')
mySquare = lambda x : x**2
print(mySquare(2))
print(mySquare(5))

mySimpleFunc = lambda x, y, z : 2*x + 3*y + z
print(mySimpleFunc(1, 2, 3), '\n')


# 내장 함수
myNum = [10, 5, 12, 0, 3.5, 99.5, 42]
print(min(myNum), max(myNum), '\n')

myStr = 'zxyabc'
print(min(myStr), max(myStr), '\n')
print("abs(-10) --> ", abs(-10))

numList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("sum(numList) --> ", sum(numList))
print("len(numList) --> ", len(numList), '\n')