# 구분자를 기준으로 문자열 분리
coffee_menu_str = "에스프레소, 아메리카노, 카페라떼, 카푸치노"
print(coffee_menu_str.split(','))
print(coffee_menu_str.split(',', maxsplit=2))


# 필요없는 문자열 삭제
# 앞 뒤 제거 --> strip
str01 = " Python "
# 공백 제거
print(str01.strip())
str02 = "aaaaaPythonbbba"
print(str02.strip('a'))
print(str02.strip('b'))
print(str02.strip('ab'))


# 구분자를 사용하여 리스트 연결
address_list = ["서울시", "서초구", "반포대로", "201(반포동)"]
separator = "*^^*"
print(separator.join(address_list))

# 문자열 찾기
str_f = "Python is powerfil. Python is easy to learn."
print("찾는 문자열의 위치:", str_f.find("Python"))
print("찾는 문자열의 위치 (10~30사이):", str_f.find("Python", 10, 30))
print("찾는 문자열의 개수:", str_f.count("Python"))

# 문자열이 지정된 문자열로 시작하는지 끝나는지
print("Python으로 시작?", str_f.startswith("Python"))
print("is으로 시작?", str_f.startswith("is"))
print(".으로 끝?", str_f.endswith("."))
print("learn으로 끝?", str_f.endswith("learn"))


# 문자열 바꾸기
print(str_f.replace('Python', 'Ipython'))

# 문자열 구성 확인하기
print('Python'.isalpha())  # 문자열에 공백, 특수문자, 숫자가 없음
print('12345'.isdigit())  # 문자열이 모두 숫자로 구성됨
print('abc123'.isalnum())  # 특수문자나 공백이 아닌 문자와 숫자로 구성됨 
print('  '.isspace())  # 문자열이 공백으로만 구성됨
print('PYTHON'.isupper())  # 문자열이 모두 대문자로 구성됨
print('python'.islower())  # 문자열이 모두 소문자로 구성됨

# 대소문자 변경
print(str_f.lower())
print(str_f.upper())