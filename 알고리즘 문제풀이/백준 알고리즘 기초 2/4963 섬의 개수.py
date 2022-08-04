import input_setting

import sys
from collections import deque

move_list = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]

def BFS(v):
    nodes = deque()
    nodes.append(v)

    while nodes:
        x, y = nodes.popleft()

        for move in move_list:
            nx = x + move[0]
            ny = y + move[1]

            if 0<=nx<cols and 0<=ny<rows:
                if board[ny][nx] == 1:
                    board[ny][nx] = 0
                    nodes.append([nx, ny])

while True:
    cols, rows = map(int, input().split())

    if cols == 0 or rows == 0:
        break

    board = [list(map(int, sys.stdin.readline().split())) for _ in range(rows)]

    count = 0
    for row in range(rows):
        for col in range(cols):
            if board[row][col] == 1:
                board[row][col] = 0
                count += 1
                BFS([col, row])

    print(count)