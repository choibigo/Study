from collections import deque
import sys

def solution(board):
    move_list = ((0,-1),(1,0),(0,1),(-1,0))
    
    rows = len(board)
    cols = len(board[0])
    
    def BFS(direct):
        nodes = deque()
        nodes.append([0,0, 0,direct])
        res = [[sys.maxsize for _ in range(cols)] for _ in range(rows)]
        res[0][0] = 0
        
        while nodes:
            x, y, c, d = nodes.popleft()
            for nd in range(4):
                nx = x + move_list[nd][0]
                ny = y + move_list[nd][1]
                
                if 0<=nx<cols and 0<=ny<rows:
                    if board[ny][nx] == 0:
                        if nd == d:
                            ncost = c + 100
                        else:
                            ncost = c + 600

                        if res[ny][nx] > ncost:
                            res[ny][nx] = ncost
                            nodes.append([nx, ny, ncost, nd])
        return res[-1][-1]
    
    return min(BFS(1), BFS(2))
    