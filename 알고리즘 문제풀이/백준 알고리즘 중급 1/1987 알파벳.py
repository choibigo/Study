import input_setting

import sys

rows, cols = map(int, input().split())

board = [list(sys.stdin.readline()) for _ in range(rows)]
move_list = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def DFS(pos, count):
    global max_count
    max_count = max(max_count, count)

    x,y = pos
    for move in move_list:
        nx = x + move[0]
        ny = y + move[1]

        if 0<=nx<cols and 0<=ny<rows:
            if res[ord(board[ny][nx]) - 65] == False:
                res[ord(board[ny][nx]) - 65] = True
                DFS([nx, ny], count+1)
                res[ord(board[ny][nx]) - 65] = False

max_count = 1
res = [False] * 26
res[ord(board[0][0]) - 65] = True
DFS([0,0], 1)

print(max_count)