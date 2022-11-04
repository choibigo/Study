from collections import deque

def solution(n, edge):
    graph = [[] for _ in range(n+1)]
    
    for a, b in edge:
        graph[a].append(b)
        graph[b].append(a)
    
    res = [-1 for _ in range(n+1)]
    res[1] = 0
    nodes = deque()
    nodes.append([1, 0])
    
    while nodes:
        v, count = nodes.popleft()
        
        for g in graph[v]:
            if res[g] == -1:
                res[g] = count+1
                nodes.append([g, count+1])
    
    return res.count(max(res))
    