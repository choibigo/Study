import input_setting

from collections import deque
import copy

n = int(input())
board = [list(input()) for _ in range(n)]
origin_ch = [[False for _ in range(n)] for _ in range(n)]
move_list = [(0, -1),(1,0),(0,1),(-1,0)]


def bfs(start, w):
    col, row = start
    nodes = deque()
    nodes.append([col, row])
    
    while nodes:
        x, y = nodes.popleft()

        for move in move_list:
            dx = x + move[0]
            dy = y + move[1]
            if 0<=dx<n and 0<=dy<n:
                if board[dy][dx] in w and not ch[dy][dx]:
                    nodes.append([dx, dy])
                    ch[dy][dx] = True

ch = copy.deepcopy(origin_ch)
count_1 = 0
for row in range(n):
    for col in range(n):
        if not ch[row][col]:
            if ch[row][col] == 0:
                ch[row][col] = True
                count_1 += 1
                bfs([col,row], [board[row][col]])


ch = copy.deepcopy(origin_ch)
count_2 = 0
for row in range(n):
    for col in range(n):
        if not ch[row][col]:
            ch[row][col] = True
            count_2 += 1
            if board[row][col] == "B":
                bfs([col,row], ["B"])
            else:
                bfs([col,row], ["R", "G"])

print(count_1, count_2)