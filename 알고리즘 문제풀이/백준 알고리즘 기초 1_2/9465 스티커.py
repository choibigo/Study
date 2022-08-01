import input_setting

import sys

case = int(input())

for _ in range(case):
    cols = int(input())
    
    board = [list(map(int, sys.stdin.readline().split())) for _ in range(2)]

    if cols == 1:
        print(max(board[0][0], board[1][0]))
        continue

    res = [[0 for _ in range(cols)] for _ in range(2)]

    res[0][0] = board[0][0]
    res[1][0] = board[1][0]
    
    res[0][1] = res[1][0] + board[0][1]
    res[1][1] = res[0][0] + board[1][1]

    for i in range(2, cols):
        res[0][i] = max([res[1][i-1], res[0][i-2], res[1][i-2]]) + board[0][i]
        res[1][i] = max([res[0][i-1], res[0][i-2], res[1][i-2]]) + board[1][i]

    print(max(map(max, res)))