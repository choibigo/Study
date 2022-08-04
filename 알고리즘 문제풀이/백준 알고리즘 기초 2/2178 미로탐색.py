import input_setting

from collections import deque 

move_list = [(0,-1),(1,0),(0,1),(-1,0)]

rows, cols = map(int, input().split())
board = [list(map(int, (str(input())))) for _ in range(rows)]
res = [[-1 for _ in range(cols)] for _ in range(rows) ]

nodes = deque()
nodes.append([0,0])
res[0][0] = 1

while nodes:
    x, y = nodes.popleft()

    for move in move_list:
        nx = move[0] + x
        ny = move[1] + y

        if 0<=nx<cols and 0<=ny<rows:

            if board[ny][nx] == 1:
                if res[ny][nx] == -1:
                    res[ny][nx] = res[y][x] + 1
                    nodes.append([nx, ny])
                else:
                    if res[ny][nx] > res[y][x] + 1:
                        res[ny][nx] = res[y][x] + 1
                        nodes.append([nx, ny])


print(res[rows-1][cols-1])