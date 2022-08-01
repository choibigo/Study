import input_setting

import sys

n = int(input())
num_list = list(map(int, sys.stdin.readline().split()))
res = [1] * n

for index in range(n):
    for j in range(index):
        if num_list[index] < num_list[j]:
            if res[index] < res[j] + 1:
                res[index] = res[j] + 1

print(max(res))