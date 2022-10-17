import input_setting

import sys
rows, cols = map(int, input().split())
start_board = [list(map(int, list(input()))) for _ in range(rows)]
target_board = [list(map(int, list(input()))) for _ in range(rows)]

count = 0
for row in range(rows-2):
    for col in range(cols-2):
        if start_board[row][col] != target_board[row][col]:
            count+=1
            for r in range(3):
                for c in range(3):
                    start_board[row+r][col+c] = start_board[row+r][col+c]^1


for start_row, target_row in zip(start_board, target_board):
    if start_row != target_row:
        print(-1)
        sys.exit()

print(count)