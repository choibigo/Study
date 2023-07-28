import input_setting

import sys
from collections import deque

move_list = [(-2, -1),(-2, 1),(0, -2),(0, 2),(2, -1),(2, 1)]

n = int(input())
start_x, start_y, end_x, end_y = map(int, input().split())
res = [[0 for _ in range(n)] for _ in range(n)]

nodes = deque()
nodes.append([start_x, start_y])


while nodes:
    x, y = nodes.popleft()

    if x == end_x and y == end_y:
        print(res[y][x])
        sys.exit()

    for move in move_list:
        nx = move[0] + x
        ny = move[1] + y

        if 0<=nx<n and 0<=ny<n:
            if res[ny][nx] == 0:
                res[ny][nx] = res[y][x] +1
                nodes.append([nx, ny])

print(-1)