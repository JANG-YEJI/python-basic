# List, Tuple, Set, Dictionary

# List
print("< List >")
print("리스트 항목 불러오기")
student1 = [90, 95, 85, 50]
print("student1 = ", student1)
print("student1[1] = ", student1[1])
print("student1[-1] = ", student1[-1], "\n")

print("리스트 항목 변경")
student1[1] = 100
print("After \'student1[1] = 100\' -->", student1, "\n")

print("리스트 연산")
list_01 = [1, 2, 3, 4]
list_02 = [5, 6, 7, 8]
list_con = list_01 + list_02
print("list01 = ", list_01)
print("list02 = ", list_02)
print("list01 + list02 = ", list_con)
list_con = list_01 * 3
print("list01 * 3 = ", list_con, "\n")

print("리스트 항목 존재여부")
list_data = [1, 2, 3, 4, 5]
print("list_data = ", list_data)
print("5 in list_data --> ", 5 in list_data)
print("6 in list_data --> ", 6 in list_data, "\n")

print("리스트 메서드")
myFriends = ['James', 'Robert', 'Lisa', 'Mary']
print("myFriends = ", myFriends)
# 추가
myFriends.append('Thomas')
print("After 'myFriends.append('Thomas')' -->", myFriends)
# 원하는 위치에 추가
myFriends.insert(1, 'Paul')
print("After 'myFriends.insert(1, 'Paul')' -->", myFriends)
# 여러개 추가
myFriends.extend(['Laura', 'Betty'])
print("After 'myFriends.extend(['Laura', 'Betty'])' -->", myFriends)
# 삭제
myFriends.remove('Paul')
print("After 'myFriends.remove(['Paul', 'Betty'])' -->", myFriends)
print("-"*40, "\n")



# Tuple
print("<Tuple>")
tuple1 = (1, 2, 3, 4)
print("tuple1 = ", tuple1)
print("'tuple1[1] = 5' 실행 시 오류발생, 튜플 항목 변경X")
print("'del tuple1[1]' 실행 시 오류발생, 튜플 항목 변경X")

tuple2 = ('a', 'a', 'a', 'b', 'b', 'c', 'd')
print("tuple2 = ", tuple2)
print("After 'tuple2.index('b')' --> ", tuple2.index('b'))
print("After 'tuple2.count('b')' --> ", tuple2.count('b'))
print("-"*40, "\n")



# Set
print("<Set>")
# 중복 허용 X
set1 = {1, 2, 3, 3}
print("(중복 허용 X)  set1 = {1, 2, 3, 3} --> set1: ", set1, '\n')

# 집합
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8, 9, 10}
print("A = ", A)
print("B = ", B)

print("A.intersection(B): ", A.intersection(B))
print("A.union(B): ", A.union(B))
print("A.difference(B): ", A.difference(B))
print("-"*40, "\n")



# Dictionary
print("<Dictionary>")
country_capital = {"영국":"런던", "프랑스":"파리", "스위스":"베른", "호주":"멜버른", "덴마크":"코펜하겐"}
print("country_capital: ", country_capital)
print("country_capital[\"영국\"] --> ", country_capital["영국"], '\n')

# 다양한 형태의 키와 값을 갖는 딕셔너리
mixed_dict = {1:10, "영국":"런던", "dict_list":[11, 12, 13], "dict_list_tuple":{"A":[11, 12, 13], "B":[21, 22, 23]}}
print("mixed_dict --> ", mixed_dict, '\n')

#데이터 추가 변경 삭제
country_capital["독일"] = "베를린"
print("After 'country_capital[\"독일\"] = \"베를린\"' --> ", country_capital)
country_capital["호주"] = "캔버라"
print("After 'country_capital[\"호주\"] = \"캔버라\"' --> ", country_capital)
del country_capital["덴마크"]
print("After 'del country_capital[\"덴마크\"]' --> ", country_capital, '\n')

#딕셔너리 메서드
print("country_capital.keys() --> ", country_capital.keys())
print("country_capital.values() --> ", country_capital.values())
print("country_capital.items() --> ", country_capital.items(), '\n')

country_capital2 = {"덴마크":"코펜하겐", "대한민국":"서울"}
country_capital.update(country_capital2)
print("country_capital2 = {\"덴마크\":\"코펜하겐\", \"대한민국\":\"서울\"} = ", country_capital2)
print("After 'country_capital.update(country_capital2)' --> ", country_capital.update(country_capital2))
