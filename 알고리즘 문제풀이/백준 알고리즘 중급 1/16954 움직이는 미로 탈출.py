import input_setting

from collections import deque


def bfs():
    board = [list(input()) for _ in range(8)]
    visited = [[False for _ in range(8)] for _ in range(8)]
    move_list = [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,0)]

    nodes = deque()
    nodes.append([0, 7])
    visited[7][0] = True

    while nodes:
        x, y= nodes.popleft()

        if board[y][x] == "#":
            continue
        if y==0:
            return 1

        for move in move_list:
            nx = x + move[0]
            ny = y + move[1]

            if 0<=nx<8 and 0<=ny<8 and board[ny][nx] != "#":
                if not visited[ny-1][nx]:
                    visited[ny-1][nx] = True
                    nodes.append([nx, ny-1])
    return 0

print(bfs())


