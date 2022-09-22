import sys
from collections import defaultdict

n = int(input())
graph = dict()
parentNode = [-1] * (n+1)
col_num = [-1] * (n+1)
depth_info = defaultdict(list)
count = 0

def DFS(v, depth):

    global count
    if graph[v][0] != -1:
        DFS(graph[v][0], depth+1)

    count +=1
    col_num[v] = count
    depth_info[depth].append(v)

    if graph[v][1] != -1:
        DFS(graph[v][1], depth+1)

for _ in range(n):
    parent, child1, child2 = map(int, sys.stdin.readline().split())
    graph[parent] = [child1, child2]

    if child1 != -1:
        parentNode[child1] = parent
    if child2 != -1:
        parentNode[child2] = parent

for i in range(1, n+1):
    if parentNode[i] == -1:
        root = i
        break

DFS(root, 1)
max_level = -1
max_width = -1
for i in range(1, len(depth_info)+ 1):
    temp = col_num[depth_info[i][-1]] - col_num[depth_info[i][0]]

    if max_width < temp:
        max_width = temp
        max_level = i

print(max_level, max_width+1)
