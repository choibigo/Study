
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
def make_graph(n, matrix):
    
    graph = [[] for _ in range(n)]
    for start in range(n):
        for end in range(n):
            if start == end:
                continue
            
            if matrix[start][end] == 1:
                graph[start].append(end)
    
    return graph

def DFS(v, graph, visited):
    for g in graph[v]:
        if visited[g] == False:
            visited[g] = True
            DFS(g, graph, visited)

def solution(n, computers):
    
    graph = make_graph(n, computers)
    visited = [False] * n

    count = 0
    
    for i in range(n):
        if visited[i] == False:
            visited[i] = True
            count +=1
            DFS(i, graph, visited)
            
    return count
# endregion
