import input_setting


import sys
from collections import deque

rows, cols = map(int, input().split())
board = [list(map(int,list(input()))) for _ in range(rows)]


def BFS(rows, cols, board):
    visited = [[[0]*2 for _ in range(cols)] for _ in range(rows)]
    visited[0][0][0] = 1
    move_list = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    nodes = deque()
    nodes.append([0,0,0])

    while nodes:
        x, y, c = nodes.popleft()

        if x==cols-1 and y==rows-1:
            return visited[y][x][c]

        for move in move_list:
            nx = x + move[0]
            ny = y + move[1]

            if 0<=nx<cols and 0<=ny<rows:
                if board[ny][nx] == 1 and c == 0:
                    visited[ny][nx][1] = visited[y][x][0] + 1
                    nodes.append([nx, ny, 1])
                elif board[ny][nx] == 0 and visited[ny][nx][c] == 0:
                    visited[ny][nx][c] = visited[y][x][c]+1
                    nodes.append([nx, ny, c])
    return -1

print(BFS(rows, cols, board))