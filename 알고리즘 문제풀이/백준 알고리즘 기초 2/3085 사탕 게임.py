import input_setting

import sys

def horizontal_count(x,y):
    
    target_char = board[y][x]
    count = 1
    for i in range(y-1, -1, -1):
        if board[i][x] == target_char:
            count +=1
        else:
            break
    
    for i in range(y+1, n, 1):
        if board[i][x] == target_char:
            count +=1
        else:
            break

    return count

def vertical_count(x,y):
    target_char = board[y][x]
    count = 1
    for i in range(x-1, -1, -1):
        if board[y][i] == target_char:
            count +=1
        else:
            break
    
    for i in range(x+1, n, 1):
        if board[y][i] == target_char:
            count +=1
        else:
            break

    return count

n = int(input())
board = [list(input()) for _ in range(n)]

move_list = [(0,-1),(1,0),(0,1),(-1,0)]
max_count = 1

for row in range(n):
    for col in range(n):
        count_1 = vertical_count(col, row)
        count_2 = horizontal_count(col, row)

        normal_count = max(count_1, count_2)

        if max_count < normal_count:
            max_count = normal_count

        for move in move_list:
            nx = move[0] + col
            ny = move[1] + row

            if 0<=nx<n and 0<=ny<n:
                if board[row][col] != board[ny][nx]:
                    if move[0] == 0:
                        board[row][col], board[ny][nx] = board[ny][nx], board[row][col]
                        count = max(vertical_count(col, row), horizontal_count(col, row))
                        board[row][col], board[ny][nx] = board[ny][nx], board[row][col]

                    elif move[1] == 0:
                        board[row][col], board[ny][nx] = board[ny][nx], board[row][col]
                        count = max(vertical_count(col, row), horizontal_count(col, row))
                        board[row][col], board[ny][nx] = board[ny][nx], board[row][col]

                    if max_count < count:
                        max_count = count

        if max_count == n:
            print(max_count)
            sys.exit()

print(max_count)
# print(*board, sep="\n")