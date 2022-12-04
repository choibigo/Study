import input_setting


import sys
board = [list(map(int, input())) for _ in range(9)]

zero_pos = list()
for row in range(9):
    for col in range(9):
        if board[row][col] == 0:
            zero_pos.append((col, row))
len_zero_count = len(zero_pos)

def DFS(idx):
    if idx == len_zero_count:
        for board_row in board:
            print(*board_row, sep='')
        sys.exit(0)

    x, y = zero_pos[idx]
    info = {i:True for i in range(0, 10)}

    # col, row
    for j in range(9):
        info[board[j][x]] = False
        info[board[y][j]] = False

    # block
    for r in range(y//3*3, y//3*3+3):
        for c in range(x//3*3, x//3*3+3):
            info[board[r][c]] = False  

    for i in range(1, 10):
        if info[i]:
            board[y][x] = i
            DFS(idx+1)
            board[y][x] = 0
DFS(0)