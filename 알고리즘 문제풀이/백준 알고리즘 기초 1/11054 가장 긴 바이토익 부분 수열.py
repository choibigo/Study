import input_setting

import sys

n = int(input())
num_list = list(map(int, sys.stdin.readline().split()))

res = [[1 for _ in range(n)] for _ in range(2)]

for index in range(n):
    for j in range(index):
        if num_list[index] > num_list[j]:
            if res[0][index] < res[0][j] + 1:
                res[0][index] = res[0][j] + 1

for index in range(n-1, -1, -1):
    for j in range(index+1, n):
        if num_list[index] > num_list[j]:
            if res[1][index] < res[1][j] + 1:
                res[1][index] = res[1][j] +1

max_value = 1
for i in range(n):
    temp = (res[0][i] + res[1][i]) - 1
    if max_value < temp:
        max_value = temp

print(max_value)