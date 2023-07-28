
# region BFS
from collections import deque

def solution(n, computers):

    res = [False]*(n)
    count = 0
    
    for i in range(n):
        if not res[i]:
            count += 1
            res[i] = True

            nodes = deque([i])
            while nodes:
                v = nodes.popleft()
                for vi in range(n):
                    if v != vi and computers[v][vi] and not res[vi]:
                        res[vi] = True
                        nodes.append(vi)
    
    return count
# endregion

# region DFS
def solution(n, computers):

    graph = [[] for _ in range(n)]
    check = [False for _ in range(n)]
    for start in range(n):
        for end in range(start+1, n):
            if computers[start][end]:
                graph[start].append(end)
                graph[end].append(start)
    
    def DFS(v, check):
        check[v] = True
        for g in graph[v]:
            if not check[g]:
                check[g] = True
                DFS(g, check)
    
    count = 0
    for i in range(n):
        if not check[i]:
            count +=1
            DFS(i, check)
            
    return count
# endregion
