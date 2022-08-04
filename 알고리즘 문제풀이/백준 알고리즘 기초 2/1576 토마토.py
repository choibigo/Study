import input_setting

import sys
from collections import deque


cols, rows = map(int, input().split())
board = [list(map(int, sys.stdin.readline().split())) for _ in range(rows)]

nodes = deque()

for row in range(rows):
    for col in range(cols):
        if board[row][col] == 1:
            nodes.append([col, row])

move_list = [(0,-1),(1,0),(0,1),(-1,0)]

while nodes:
    x, y = nodes.popleft()

    for move in move_list:
        nx = x + move[0]
        ny = y + move[1]
    
        if 0<=nx<cols and 0<=ny<rows:
            if board[ny][nx] == 0:
                board[ny][nx] = board[y][x] + 1
                nodes.append([nx, ny]) 

max_count = 0
for row in range(rows):
    for col in range(cols):
        if board[row][col] == 0:
            print(-1)
            sys.exit()

        if max_count < board[row][col]:
            max_count = board[row][col]

print(max_count - 1)