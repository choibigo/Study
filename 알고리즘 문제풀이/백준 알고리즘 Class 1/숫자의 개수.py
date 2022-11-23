import input_setting


from collections import Counter

a = int(input())
b = int(input())
c = int(input())

num = a*b*c
info = Counter(str(num))

for i in range(10):
    print(info[str(i)])