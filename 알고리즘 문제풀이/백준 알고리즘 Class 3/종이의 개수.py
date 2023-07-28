import input_setting


import sys
size = int(sys.stdin.readline())
board = [list(map(int, sys.stdin.readline().split(" "))) for _ in range(size)]
answer = [0,0,0]

def DFS(pos, size):
    x, y = pos
    start = board[y][x]

    for row in range(y, y+size):
        for col in range(x, x+size):
            if board[row][col] != start:
                DFS((x,y), size//3)
                DFS((x+size//3,y), size//3)
                DFS((x+(size//3)*2,y), size//3)
                DFS((x,y+size//3), size//3)
                DFS((x+size//3,y+size//3), size//3)
                DFS((x+(size//3)*2,y+size//3), size//3)
                DFS((x,y+(size//3)*2), size//3)
                DFS((x+size//3,y+(size//3)*2), size//3)
                DFS((x+(size//3)*2,y+(size//3)*2), size//3)
                return 
    
    answer[start+1] +=1

DFS((0,0), size)
print(*answer, sep="\n")