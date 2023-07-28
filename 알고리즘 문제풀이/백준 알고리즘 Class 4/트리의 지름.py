import input_setting


import sys
from collections import deque

n = int(input())
graph = [[] for _ in range(n+1)]
while True:
    try:
        a, b, cost = map(int, sys.stdin.readline().split())
        graph[a].append([b, cost])
        graph[b].append([a, cost])
    except:
        break

def nodes_distance(v):
    res = [-1 for _ in range(n+1)]
    
    nodes = deque([v])
    res[v] = 0

    while nodes:
        v = nodes.popleft()

        for g, cost in graph[v]:
            if res[g] == -1:
                res[g] = res[v]+cost
                nodes.append(g)
    return res

distance_1 = nodes_distance(1)
max_value = max(distance_1)

result = 0
for idx, r in enumerate(distance_1):
    if r == max_value:
        result = max(result, max(nodes_distance(idx)))

print(result)
