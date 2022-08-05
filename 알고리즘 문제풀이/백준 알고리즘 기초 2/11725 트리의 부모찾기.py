import input_setting

import sys
sys.setrecursionlimit(10 ** 6)

def DFS(v):

    for g in graph[v]:
        if res[g] == -1:
            res[g] = v
            DFS(g)


n = int(input())
graph = [[] for _ in range(n+1)]

for _ in range(n-1):
    a, b = map(int, input().split())

    graph[a].append(b)
    graph[b].append(a)


res = [-1] * (n+1)
res[1] = 0
DFS(1)

print(*(res[2:]), sep="\n")