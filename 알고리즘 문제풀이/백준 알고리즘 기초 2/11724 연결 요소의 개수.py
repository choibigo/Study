import input_setting

import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

def DFS(v):
    visited[v] = 1
    for g in graph[v]:
        if visited[g] == 0:
            DFS(g)


node_count, edge_count = map(int, input().split())
graph = [[] for _ in range(node_count+1)]

for _ in range(edge_count):
    a, b = map(int, input().split())

    graph[a].append(b)
    graph[b].append(a)

visited = [0] * (node_count + 1)
ch = [0] * (node_count + 1)

count = 0 
for i in range(1, node_count + 1):
    if visited[i] == 0:
        if len(graph[i]) == 0:
            count +=1
            visited[i] = 1
        else:
            count +=1
            DFS(i)

print(count)
