import input_setting

from collections import deque

friend_count, graph_count = map(int, input().split(" "))
graph= [[] for _ in range(friend_count+1)]
init_res = [float("inf") for _ in range(friend_count+1)]

for _ in range(graph_count):
    a, b = map(int, input().split(" "))
    graph[a].append(b)
    graph[b].append(a)

min_count = float("inf")
answer = -1

for start in range(1, friend_count+1):
    res = init_res[:]
    res[start] = 0
    nodes = deque([[start, 0]])

    while nodes:
        i, count = nodes.popleft()
        for g in graph[i]:
            if res[g] > count+1:
                res[g] = count+1
                nodes.append([g, count+1])
    
    temp = sum(res[1:])
    if min_count > temp:
        min_count = temp
        answer = start

print(answer)