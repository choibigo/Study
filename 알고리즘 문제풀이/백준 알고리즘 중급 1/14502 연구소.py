import input_setting

import sys
from collections import deque
import copy

rows, cols = map(int, input().split())
origin_board = [list(map(int, sys.stdin.readline().split())) for _ in range(rows)]
virus_pos = deque()
move_list = [(0, 1), (0, -1), (1, 0), (-1, 0)]

for row in range(rows):
    for col in range(cols):
        if origin_board[row][col] == 2:
            virus_pos.append([col, row])

max_count = 0

def bfs():
    
    board = copy.deepcopy(origin_board)
    nodes = copy.deepcopy(virus_pos)

    while nodes:
        x, y = nodes.popleft()

        for move in move_list:
            nx = x + move[0]
            ny = y + move[1]

            if 0<=nx<cols and 0<=ny<rows:
                if board[ny][nx] == 0:
                    board[ny][nx] = 1
                    nodes.append([nx, ny])

    count = 0
    for row in range(rows):
        for col in range(cols):
            if board[row][col] == 0:
                count += 1

    return count


def wall(count):
    global max_count

    if count == 3:
        max_count= max(bfs(), max_count)
        return 

    for row in range(rows):
        for col in range(cols):
            if origin_board[row][col] == 0:
                origin_board[row][col] = 1
                wall(count+1)
                origin_board[row][col] = 0
wall(0)
print(max_count)
