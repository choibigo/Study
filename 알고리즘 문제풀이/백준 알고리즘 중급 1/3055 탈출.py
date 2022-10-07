import input_setting

from collections import deque

def bfs():
    rows, cols = map(int, input().split())
    board = [list(input()) for _ in range(rows)]
    res = [[0 for _ in range(cols)] for _ in range(rows)]
    move_list = [(-1,0),(1,0),(0,1),(0,-1)]

    nodes = deque()
    temp = list()
    for row in range(rows):
        for col in range(cols):
            if board[row][col] == "D":
                dx = col
                dy = row
            if board[row][col] == "S":
                nodes.append([col, row])
            if board[row][col] == "*":
                temp.append([col, row])

    for t in temp:
        nodes.append(t)

    while nodes:
        x, y = nodes.popleft()
        if x == dx and y == dy:
            return res[y][x]

        for move in move_list:
            nx = x + move[0]
            ny = y + move[1]

            if 0<=nx<cols and 0<=ny<rows:
                if board[ny][nx] != "X":
                    if board[y][x] == "S" and (board[ny][nx] == "." or board[ny][nx]=='D'):
                        board[ny][nx] = "S"
                        res[ny][nx] = res[y][x] + 1
                        nodes.append([nx, ny])

                    elif board[y][x] == "*" and (board[ny][nx] == "." or board[ny][nx]=='S'):
                        board[ny][nx] = "*"
                        nodes.append([nx, ny])

    return "KAKTUS"

print(bfs())