import input_setting

import sys
board = [list(map(int, input().split())) for _ in range(9)]

blank_list = list()
for row in range(9):
    for col in range(9):
        if board[row][col] == 0:
            blank_list.append([col, row])

def check_row(y, num):
    for i in range(9):
        if board[y][i] == num:
            return False
    return True

def check_col(x, num):
    for i in range(9):
        if board[i][x] == num:
            return False
    return True

def check_rect(x, y, num):
    rx = x // 3 * 3
    ry = y // 3 * 3

    for r in range(3):
        for c in range(3):
            if board[ry + r][rx + c] == num:
                return False
    return True


def DFS(v):
    if v == len(blank_list):
        for board_row in board:
            print(*board_row)

        sys.exit(0)
    
    for i in range(1, 10):
        x = blank_list[v][0]
        y = blank_list[v][1]

        if check_col(x, i) and check_row(y, i) and check_rect(x, y, i):
            board[y][x] = i
            DFS(v+1)
            board[y][x] = 0

DFS(0)