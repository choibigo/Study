import input_setting

size = int(input())
board = [list(map(int, input().split(" "))) for _ in range(size)]
answer = [0,0]

def DFS(pos, size):
    x, y = pos
    start = board[y][x]

    for row in range(y, y+size):
        for col in range(x, x+size):
            if board[row][col] != start:
                DFS((x,y), size//2)
                DFS((x+size//2,y), size//2)
                DFS((x,y+size//2), size//2)
                DFS((x+size//2,y+size//2), size//2)
                return 
    
    answer[start]+=1
DFS((0,0), size)
print(answer[0])
print(answer[1])
    