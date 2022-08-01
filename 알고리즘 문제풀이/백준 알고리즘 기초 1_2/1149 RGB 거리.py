import input_setting

import sys

n = int(input())

board = [list(map(int, sys.stdin.readline().split())) for _ in range(n)]
res = [[0 for _ in range(3)] for _ in range(n)]
res[0] = board[0]

for i in range(1, n):
    for rgb in range(3):
        if rgb == 0:
            temp = [res[i-1][1], res[i-1][2]]
        elif rgb == 1:
            temp = [res[i-1][0], res[i-1][2]]
        elif rgb == 2:
            temp = [res[i-1][0], res[i-1][1]]

        res[i][rgb] = min(temp) + board[i][rgb]

print(min(res[n-1]))
