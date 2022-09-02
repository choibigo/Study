from collections import deque

def BFS(start, graph, check):
    nodes = deque()
    nodes.append(start)
    
    check[start] = True
    count = 1
    while nodes:
        pop_v = nodes.popleft()
        
        for g in graph[pop_v]:
            if not check[g]:
                check[g] = True
                nodes.append(g)
                count +=1
        
    return count
                
def solution(n, wires):
    
    check = [False for _ in range(n+1)]
    graph = [[] for _ in range(n+1)]
    answer = n
    
    for wire in wires:
        a, b = wire
        graph[a].append(b)
        graph[b].append(a)
    
    for wire in wires:
        start1, start2 = wire
        
        check[start1] = True
        check[start2] = True
        
        count1 = BFS(start1, graph, check[:])
        count2 = BFS(start2, graph, check[:])
        
        answer = min(answer, abs(count1-count2))
        
        check[start1] = False
        check[start2] = False
        
    
    return answer