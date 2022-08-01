import input_setting

import sys

n = int(input())

board = [list(map(int, input().split())) for _ in range(n)]
result = list()

for color in range(3):
    res = [[0 for _ in range(3)] for _ in range(n)]
    res[0] = [sys.maxsize,  sys.maxsize, sys.maxsize]
    res[0][color] = board[0][color]

    for i in range(1, n):
        res[i][0] = min([res[i-1][1], res[i-1][2]]) + board[i][0]
        res[i][1] = min([res[i-1][0], res[i-1][2]]) + board[i][1]
        res[i][2] = min([res[i-1][1], res[i-1][0]]) + board[i][2]

    res[n-1][color] = sys.maxsize
    result.append(min(res[n-1]))

print(min(result))
    




