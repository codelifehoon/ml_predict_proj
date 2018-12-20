from functools import reduce


b = {"c":90, "b":45, "a": 88}

print(sorted(b.items()))


# 예제2 - zip()을 이용한 Dictionary의 value기준으로
# 최소(min), 최대(max)값 찾기 및 정렬하기
# 1) 최소, 최대값 찾기

d = {'banana': 3, 'apple': 4, 'pear': 1, 'orange':2}

min_item = min(zip(d.values(), d.keys()))
max_item = max(zip(d.values(), d.keys()))
print(min_item)
print(max_item)


# 2) 오름차순으로 정렬하기
d_sorted = sorted(zip(*d))
print(d_sorted)

d_sorted2 = sorted(d.items(), key= lambda t:t[1])






primes = [2, 3, 5, 7, 11, 13]

def product(*numbers):
    print(numbers)

product(*primes) # => product(2, 3, 5, 7, 11, 13)  6개의 param이 전달된 형채
# 30030

product(primes)  #=> product([2, 3, 5, 7, 11, 13]) 이런 형태의 하나의 param이 전달된형태

