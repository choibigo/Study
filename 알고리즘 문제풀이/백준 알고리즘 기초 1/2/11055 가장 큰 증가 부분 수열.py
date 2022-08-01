import input_setting

import sys
import copy

n = int(input())
num_list = list(map(int, sys.stdin.readline().split()))
res = copy.deepcopy(num_list)

if n == 1:
    print(num_list[0])
    sys.exit()


for target_index in range(1, n):
    for j in range(target_index):
        if num_list[j] < num_list[target_index]:
            if res[target_index] < res[j] + num_list[target_index]:
                res[target_index] = res[j] + num_list[target_index]

print(max(res))