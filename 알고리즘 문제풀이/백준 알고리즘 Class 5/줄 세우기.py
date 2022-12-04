import input_setting



from collections import deque
node_count, path_count = map(int, input().split())
graph = [[] for _ in range(node_count+1)]
link_count = [0 for _ in range(node_count+1)]

for _ in range(path_count):
    a, b = map(int, input().split())
    graph[a].append(b)
    link_count[b] +=1

nodes = deque()
for v in range(1, node_count+1):
    if link_count[v] == 0:
        nodes.append(v)

while nodes:
    current_v = nodes.popleft()
    print(current_v, end=' ')
    for g in graph[current_v]:
        link_count[g] -=1

        if link_count[g] == 0:
            nodes.append(g)