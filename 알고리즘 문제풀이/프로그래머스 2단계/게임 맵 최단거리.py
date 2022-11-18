from collections import deque

def solution(maps):
    
    rows = len(maps)
    cols = len(maps[0])
    
    move_list = [(0,-1),(1,0),(0,1),(-1,0)]
    res = [[0 for _ in range(cols)] for _ in range(rows)]
    
    nodes = deque()
    nodes.append([0,0])
    maps[0][0] = 1
    res[0][0] = 1

    while nodes:
        x, y = nodes.popleft()
        
        if x == cols-1 and y == rows-1:
            return res[rows-1][cols-1]
        
        for move in move_list:
            nx = move[0] + x    
            ny = move[1] + y    
    
            if 0<=nx<cols and 0<=ny<rows:
                if maps[ny][nx] == 1:
                    maps[ny][nx] = 0
                    res[ny][nx] = res[y][x] + 1
                    nodes.append([nx, ny])
    return -1