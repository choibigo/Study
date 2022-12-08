from collections import deque

def solution(n, roads, sources, destination):

    path = [-1 for _ in range(n+1)]
    graph = [[] for _ in range(n+1)]
    for a, b in roads:
        graph[a].append(b)
        graph[b].append(a)
        
    nodes = deque([destination])
    path[destination] = 0
    
    while nodes:
        v = nodes.popleft()
        for g in graph[v]:
            if path[g] == -1:
                path[g] = path[v]+1
                nodes.append(g)
                
    return [path[s] for s in sources]
