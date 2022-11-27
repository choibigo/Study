import input_setting


import sys
n = int(input())
graph = [list(map(int, sys.stdin.readline().split(' '))) for _ in range(n)]

for k in range(n):
    for i in range(n):
        for j in range(n):
            if graph[i][k] and graph[k][j]:
                graph[i][j] = 1

for g in graph:
    print(*g)