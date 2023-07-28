import input_setting

import sys
n = int(input())
first_row = list(map(int, sys.stdin.readline().split(' ')))
res_max = first_row[:]
res_min = first_row[:]

for _ in range(n-1):
    temp = list(map(int, sys.stdin.readline().split(' ')))
    temp_max = res_max[:]
    temp_min = res_min[:]

    for col in range(3):
        if col == 0:
            res_max[col] = max(temp_max[col], temp_max[col+1]) + temp[col]
            res_min[col] = min(temp_min[col], temp_min[col+1]) + temp[col]
        elif col == 2:
            res_max[col] = max(temp_max[col], temp_max[col-1]) + temp[col]
            res_min[col] = min(temp_min[col], temp_min[col-1]) + temp[col]
        else:
            res_max[col] = max(temp_max[col], temp_max[col-1], temp_max[col+1]) + temp[col]
            res_min[col] = min(temp_min[col], temp_min[col-1], temp_min[col+1]) + temp[col]

print(max(res_max), min(res_min))