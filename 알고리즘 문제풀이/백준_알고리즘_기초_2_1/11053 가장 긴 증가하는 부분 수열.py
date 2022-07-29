import sys
import copy 

import input_setting # Need To Delete

n = int(input())
num_list = list(map(int, sys.stdin.readline().split()))

res = [1] * (n)

for index in range(1, n):
    for j in range(0, index):
        if num_list[index] > num_list[j] and res[index] < res[j] + 1:
            res[index] = res[j] + 1

max_value = max(res)
print(max_value)

