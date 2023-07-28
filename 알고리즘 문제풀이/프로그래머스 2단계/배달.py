import sys

def get_smallesxt_node(distance, visited):
    
    min_val = sys.maxsize
    index = -1
    
    for i, (d, v) in enumerate(zip(distance, visited)):
        if v == False:
            if min_val > d:
                min_val = d
                index = i
    return index

def dijkstra(start, N, road_list):
    INF = sys.maxsize-1
    visited = [False] * (N+1)
    distance = [INF] * (N+1)
    
    # 그래프 구하기
    graph = [[] for _ in range(N+1)]
    for key, w in road_list.items():
        s,e = key.split("-")
        graph[int(s)].append((int(e), w))
        graph[int(e)].append((int(s), w))
    
    print(graph)
    
    # start 초기화
    distance[start] = 0
    visited[start] = True
    visited[0] = True
    
    # 시작 노드로 distance 초기화
    for g in graph[start]:
        distance[g[0]] = g[1]
    
    for _ in range(N-1): # 전체 노드개수 - 1 만큼 반복
        # 현재 방문하지 않은 노드중 최소 값 구하기
        now = get_smallesxt_node(distance[:], visited[:])
        visited[now] = True
        
        for j in graph[now]:
            if distance[now] + j[1] < distance[j[0]]:
                distance[j[0]] = distance[now] + j[1]
        
    return distance

        
def solution(N, road_list, K):
    
    res = dict()
    
    for road in road_list:
        s, e, w = road
        key = str(min(s, e))+"-"+str(max(s,e))
        if key in res:
            if res[key] > w:
                res[key] = w
        else:
            res[key] = w

    distance = dijkstra(1, N, res)
    
    result = [1 for d in distance if d <= K]
    
    return len(result)