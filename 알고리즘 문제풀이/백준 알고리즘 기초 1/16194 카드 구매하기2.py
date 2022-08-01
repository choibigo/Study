import sys

import input_setting # Need To Delete

n = int(input())
num_list = list(map(int, sys.stdin.readline().split()))

for i in range(1, n):
    for j in range(0, i):
        num_list[i] = min(num_list[i], (num_list[j] + num_list[i-j-1]))

print(num_list[n-1])