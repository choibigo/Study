import input_setting

import sys

n = int(input())
num_list = list()
res = list()

for _ in range(n):
    num_row = list(map(int, sys.stdin.readline().split()))
    num_list.append(num_row)
    
    temp = [0] * len(num_row)
    res.append(temp)

res[0][0] = num_list[0][0]

for row in range(1, n):
    for col in range(0, row+1):
        if col == 0:
            res[row][col] = res[row-1][col] + num_list[row][col]

        elif col == row:
            res[row][col] = res[row-1][col-1] + num_list[row][col]

        else:
            temp = [res[row-1][col-1], res[row-1][col]]
            res[row][col] = max(temp) + num_list[row][col]

print(max(res[n-1]))
