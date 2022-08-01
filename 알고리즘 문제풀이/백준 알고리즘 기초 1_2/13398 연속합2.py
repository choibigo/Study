import input_setting

import sys

n = int(input())
num_list = list(map(int, sys.stdin.readline().split()))

if n == 1:
    print(num_list[0])
    sys.exit()

res1 = [0] * n
res2 = [0] * n

res1[0] = num_list[0]

for i in range(1, n):
    res1[i] = max(res1[i-1] + num_list[i], num_list[i])

res2[0] = res1[0]
res2[1] = res1[1]

for i in range(2, n):
    res2[i] = max(res1[i-2] + num_list[i], res1[i], res2[i-1] + num_list[i])

print(max(max(res1), max(res2)))