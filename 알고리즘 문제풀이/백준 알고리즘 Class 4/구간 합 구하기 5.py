import input_setting

import sys
n, test_case = map(int, input().split())
board = [list(map(int, sys.stdin.readline().split(' '))) for _ in range(n)]
res = [[0 for _ in range(n)] for _ in range(n)]
res[0][0] = board[0][0]

for c in range(1, n):
    res[0][c] = res[0][c-1]+board[0][c]

for row in range(1, n):
    for col in range(n):
        if col==0:
            res[row][col] = res[row-1][col] + board[row][col]
        else:
            res[row][col] = res[row-1][col] + res[row][col-1] - res[row-1][col-1] + board[row][col]


for _ in range(test_case):
    y1, x1, y2, x2 = map(lambda x: int(x)-1, sys.stdin.readline().split(' '))
    if x1==0 and y1==0:
        print(res[y2][x2])
    elif x1==0: 
        print(res[y2][x2] - res[y1-1][x2])
    elif y1==0:
        print(res[y2][x2] - res[y2][x1-1])
    else:
        print(res[y2][x2] - res[y2][x1-1] - res[y1-1][x2]+res[y1-1][x1-1])
