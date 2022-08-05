import input_setting

from collections import deque

case = int(input())

move_list = [(1,-2),(2,-1),(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2)]

def bfs():
    res = [[0 for _ in range(n)] for _ in range(n)]

    nodes = deque()
    nodes.append([start_pos, 0])
    res[start_pos[1]][start_pos[0]] = 1

    while nodes:
        pos, count = nodes.popleft()
        x, y = pos
        
        if x == end_pos[0] and y == end_pos[1]:
            return count

        for move in move_list:
            nx = move[0] + x
            ny = move[1] + y

            if 0<=nx<n and 0<=ny<n:
                if res[ny][nx] == 0:
                    res[ny][nx] = 1
                    nodes.append([[nx, ny],count+1])

    return -1

for _ in range(case):
    n = int(input())
    start_pos = list(map(int, input().split()))
    end_pos = list(map(int, input().split()))

    print(bfs())

