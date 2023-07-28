import input_setting


import sys
from collections import deque
rows, cols = map(int, input().split())
board = [list(sys.stdin.readline().strip()) for _ in range(rows)]
res = [[False for _ in range(cols)] for _ in range(rows)]

come_info = list()
come_info.append([0,1,'U'])
come_info.append([1,0,'L'])
come_info.append([0,-1,'D'])
come_info.append([-1,0,'R'])

next_info = dict()
next_info['U'] = (0,-1)
next_info['L'] = (-1,0)
next_info['D'] = (0,1)
next_info['R'] = (1,0)

count = 0
for row in range(rows):
    for col in range(cols):
        if not res[row][col]:
            count +=1
            res[row][col] = True
            nodes = deque([[col, row]])

            while nodes:
                x, y = nodes.popleft()

                # come
                for x_off, y_off, d in come_info:
                    nx = x+x_off
                    ny = y+y_off

                    if 0<=nx<cols and 0<=ny<rows:
                        if not res[ny][nx] and board[ny][nx] == d:
                            res[ny][nx] = True
                            nodes.append([nx, ny])

                
                nx = x+next_info[board[y][x]][0]
                ny = y+next_info[board[y][x]][1]

                if not res[ny][nx]:
                    res[ny][nx] = True
                    nodes.append([nx, ny])

print(count)