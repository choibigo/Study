import input_setting

import sys
sys.setrecursionlimit(10 ** 6)

def DFS(v):
    if len(res) == 4:
        print(1)
        sys.exit()

    for g in graph[v]:
        if ch[g] == 0:
            ch[g] = 1
            res.append(g)
            DFS(g)
            ch[g] = 0
            res.pop()

node_count, edge_count = map(int, input().split())

graph = [[] for _ in range(node_count)]

for _ in range(edge_count):
    a,b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)


for i in range(node_count):
    ch = [0] * (node_count)
    res = list()
    ch[i] = 1
    DFS(i)

print(0)