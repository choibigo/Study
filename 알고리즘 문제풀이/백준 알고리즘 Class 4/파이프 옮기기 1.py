import input_setting


import sys
n = int(input())
board = [list(map(int, sys.stdin.readline().split(' '))) for _ in range(n)]

answer = 0
def DFS(pos, direct):
    global answer
    x, y = pos
    if x==n-1 and y==n-1:
        answer +=1
        return

    if x+1<n and y+1 <n:
        if board[y+1][x+1] == 0 and board[y+1][x] == 0 and board[y][x+1] == 0:
            DFS((x+1, y+1), 2)
    
    # 가로
    if direct==0 or direct==2:
        if x+1<n and board[y][x+1] == 0:
            DFS((x+1, y), 0)
    
    # 세로
    if direct==1 or direct==2:
        if y+1<n and board[y+1][x] == 0:
            DFS((x, y+1), 1)


DFS((1,0), 0)
print(answer)
                      
