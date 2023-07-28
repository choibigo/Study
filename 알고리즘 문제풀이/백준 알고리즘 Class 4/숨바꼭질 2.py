import input_setting

from collections import deque
n, k = map(int, input().split())
visited = [float("inf") for _ in range(100002)]
check = [0 for _ in range(100002)]

nodes = deque([n])
visited[n] = 0
check[n] = 1

while nodes:
    v = nodes.popleft()
    for nv in [v-1, v+1, v*2]:
        if 0<=nv<=100000 and visited[v]+1 <= visited[nv]:
            visited[nv] = visited[v]+1
            check[nv] += 1
            nodes.append(nv)

print(visited[k])
print(check[k])