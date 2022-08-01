import sys
import copy

import input_setting # Need To Delete

n = int(input())
num_list = list(map(int, sys.stdin.readline().split()))
res = copy.deepcopy(num_list)

for i in range(1, n):
    for j in range(0, i):
        res[i] = max(res[i], (res[j] + res[i-j-1]))

print(res[n-1])

