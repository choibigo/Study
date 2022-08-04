import input_setting

import sys

move_list = [(0,-1),(1,0),(0,1),(-1,0)]

def DFS(v):

    global count

    x, y = v

    for move in move_list:
        nx = move[0] + x
        ny = move[1] + y

        if 0<=nx<n and 0<=ny<n:
            if board[ny][nx] == 1:
                board[ny][nx] = 0
                count += 1
                DFS([nx, ny])

n = int(input())
board = [list(map(int, list(str(input())))) for _ in range(n)]

total_count = 0
count_list = list()

for row in range(n):
    for col in range(n):
        if board[row][col] == 1:
            total_count += 1
            count = 1
            board[row][col] = 0
            DFS([col, row])
            count_list.append(count)

print(total_count)
count_list.sort()
print(*count_list, sep="\n")