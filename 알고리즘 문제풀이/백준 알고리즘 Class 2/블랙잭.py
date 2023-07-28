import input_setting

import sys
from itertools import combinations
n, m = map(int, sys.stdin.readline().split(' '))
num_list = list(map(int, sys.stdin.readline().split(' ')))

answer = -1
for candidate in combinations(num_list, 3):
    sum_val = sum(candidate)
    if sum_val <= m and answer < sum_val:
        answer = sum_val

print(answer)