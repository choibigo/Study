import sys

import input_setting # Need To Delete

n = int(input())
num_list = list(map(int, sys.stdin.readline().split()))

res = [0] * n
res[0] = num_list[0]

for i in range(1, n):
    res[i] = max((res[i-1] + num_list[i]), num_list[i])

print(max(res))