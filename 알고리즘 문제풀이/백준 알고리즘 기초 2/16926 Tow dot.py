import input_setting

import sys

move_list = [(0,-1),(1,0),(0,1),(-1,0)]

def DFS(v, path):
    x, y = v

    for move in move_list:
        nx = move[0] + x
        ny = move[1] + y

        if 0<=nx<cols and 0<=ny<rows:
            if char == board[ny][nx]:
                if res[ny][nx] == 0:
                    res[ny][nx] = 1
                    DFS([nx, ny], path+1)
                    res[ny][nx] = 0

                else:
                    if start_point[0] == nx and start_point[1] == ny and path >= 4:
                        print(f"Yes")
                        sys.exit()
                

rows, cols = map(int, input().split())

board = [list(input()) for _ in range(rows)]
res = [[0 for _ in range(cols)] for _ in range(rows)]

for row in range(rows):
    for col in range(cols):
        res[row][col] = 1
        char = board[row][col]
        start_point = (col, row)
        DFS([col, row], 1)

print("No")