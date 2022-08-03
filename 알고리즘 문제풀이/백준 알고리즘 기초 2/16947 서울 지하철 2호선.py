import input_setting

import sys
import copy
from collections import deque

sys.setrecursionlimit(10 ** 6)
            
def cycle(v):

    global find_cylce
    global cycle_path

    if find_cylce == True:
        return 

    for g in graph[v]:
        if res[g] == 0:
            res[g] = 1
            path.append(g)
            cycle(g)
            path.pop()
            res[g] = 0

        else:
            if start_node == g and len(path) >=3:
                cycle_path = copy.deepcopy(path)
                find_cylce = True
                return 

node_count = int(input())
graph = [[] for _ in range(node_count + 1)]

for _ in range(node_count):
    a, b = map(int, sys.stdin.readline().split())

    graph[a].append(b)
    graph[b].append(a)

res = [0] * (node_count + 1)

for i in range(1, node_count):
    path = [i]
    res[i] = 1
    start_node = i
    find_cylce = False
    cycle(i)

    if find_cylce == True:
        break


b_path_count = [-1] * (node_count + 1)
nodes = deque()

for c in cycle_path:
    nodes.append(c)
    b_path_count[c] = 0

while nodes:
    cnode = nodes.popleft()

    for g in graph[cnode]:
        if b_path_count[g] == -1:
            nodes.append(g)
            b_path_count[g] = b_path_count[cnode] + 1

print(*b_path_count[1:])