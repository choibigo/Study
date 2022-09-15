def solution(n, vertex):
    
    graph = [[] for _ in range(n+1)]
    
    for a,b in vertex:
        graph[a].append(b)
        graph[b].append(a)
    
    count_list = [-1] * (n+1)
    visited = [False] * (n+1)
    visited[1] = True
    nodes = [(1,0)]
    
    while nodes:
        pop_node, pop_count = nodes.pop(0)
        
        for g in graph[pop_node]:
            if visited[g] == False:
                visited[g] = True
                nodes.append((g, pop_count+1))
                count_list[g] = pop_count+1
                
    return count_list.count(max(count_list))