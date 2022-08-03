import input_setting

import sys
from collections import deque

def DFS(v):
    print(v, end=" ")
    for g in graph[v]:
        if ch[g] == 0:
            ch[g] = 1
            DFS(g)

def BFS(start):
    ch = [0] * (node_count + 1)
    ch[start] = 1

    nodes = deque()
    nodes.append(start)

    while nodes:
        pop_n = nodes.popleft()
        print(pop_n, end=" ")
        for g in graph[pop_n]:
            if ch[g] == 0:
                ch[g] = 1
                nodes.append(g)

node_count, edge_count, start = map(int, input().split())
graph = [[] for _ in range(node_count+1)]

for _ in range(edge_count):
    a, b = map(int, input().split())

    graph[a].append(b)
    graph[b].append(a)

for i in range(1, len(graph)):
    graph[i].sort()

ch = [0] * (node_count + 1)
ch[start] = 1
DFS(start)
print()
BFS(start)
