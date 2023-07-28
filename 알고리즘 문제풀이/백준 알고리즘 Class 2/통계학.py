import input_setting

from collections import Counter
n = int(input())
num_list = [int(input()) for _ in range(n)]
num_list.sort()
info = sorted(Counter(num_list).items(), key=lambda x : (-x[1], x[0]))

print(int(round(sum(num_list)/n)))
print(num_list[n//2])

if len(info) > 1 and info[0][1] == info[1][1]:
    print(info[1][0])
else:
    print(info[0][0])

print(num_list[-1]-num_list[0])