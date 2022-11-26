import input_setting

import sys
num_count, case = map(int, input().split())
num_list = list(map(int, sys.stdin.readline().strip().split(' ')))

res = [0] * (num_count+1)
res[1] = num_list[0]
for i in range(1, num_count):
    res[i+1] = res[i]+num_list[i]

for _ in range(case):
    a, b = map(int, sys.stdin.readline().split(' '))
    print(res[b]- res[a-1])