import input_setting

from collections import deque
import sys

rows, cols = map(int ,input().split())
board = [list(input()) for _ in range(rows)]
visited = [[True for _ in range(cols)] for _ in range(rows)]
move_list = [(0, -1), (1, 0), (0, 1), (-1, 0)]


ms = list()

for row in range(rows):
    for col in range(cols):
        if board[row][col] == "o":
            ms.append([col, row])

        if board[row][col] == ".":
            visited[row][col] = False


nodes = deque()
nodes.append([ms, 0])

while nodes:
    marbles, count = nodes.popleft()
    m1 = marbles[0]
    m2 = marbles[1]

    if count > 9:
        print(-1)
        sys.exit()

    for move in move_list:
        nm1x = m1[0] + move[0]
        nm1y = m1[1] + move[1]

        nm2x = m2[0] + move[0]
        nm2y = m2[1] + move[1]
        
        m1_condition = (0<=nm1x<cols) and (0<=nm1y<rows)
        m2_condition = (0<=nm2x<cols) and (0<=nm2y<rows)

        # 둘다 안에
        if m1_condition and m2_condition:
            
            # 둘다 가본대가 아닐때
            if not (visited[nm1y][nm1x] and visited[nm2y][nm2x]):
                temp1 = [nm1x, nm1y]
                if board[nm1y][nm1x] == "#":
                    temp1 = m1
                
                temp2 = [nm2x, nm2y]
                if board[nm2y][nm2x] == "#":
                    temp2 = m2

                nodes.append([[temp1, temp2], count+1])

        # 둘다 밖으로
        elif not (m1_condition and m2_condition) :
            pass

        # 둘중 하나만 밖으로
        if m1_condition != m2_condition:
            print(count+1)
            sys.exit()

print(-1)