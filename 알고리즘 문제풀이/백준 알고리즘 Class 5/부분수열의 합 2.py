import input_setting


# Need To Check
import bisect
from itertools import combinations
def get_sum(arr):
    sumArr = list()
    for i in range(1, len(arr) + 1):
        for a in combinations(arr, i):
            sumArr.append(sum(a))
    sumArr.sort()
    return sumArr

def get_find(arr, value):
    return bisect.bisect_right(arr, value) - bisect.bisect_left(arr, value)

import sys
n, s = map(int, input().split())
num_list = list(map(int, sys.stdin.readline().split()))
left_list, right_list = num_list[:n//2], num_list[n//2:]
left_sum_list = get_sum(left_list)
right_sum_list = get_sum(right_list)

answer = 0
for l in left_sum_list:
    answer += get_find(right_sum_list, s-l)

answer += get_find(left_sum_list, s)
answer += get_find(right_sum_list, s)
print(answer)