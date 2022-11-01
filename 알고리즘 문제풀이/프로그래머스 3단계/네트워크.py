
# region BFS
from collections import deque

def make_graph(n, matrix):
    
    graph = [[] for _ in range(n)]
    for start in range(n):
        for end in range(n):
            if start == end:
                continue
            
            if matrix[start][end] == 1:
                graph[start].append(end)
    
    return graph


def solution(n, computers):
    
    graph = make_graph(n, computers)
    
    nodes = deque()
    visited = [False] * n

    count = 0
    
    for i in range(n):
        if visited[i] ==  False:
            nodes.append(i)
            visited[0] = True
            count += 1
            
            while nodes:
                pop_node = nodes.popleft()

                for g in graph[pop_node]:
                    if visited[g] == False:
                        visited[g] = True
                        nodes.append(g)
    
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
