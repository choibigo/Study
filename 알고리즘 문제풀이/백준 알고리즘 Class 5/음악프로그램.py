import input_setting


import sys
from collections import deque
singer_count, pd_count = map(int, input().split())
graph = [[] for _ in range(singer_count+1)]
line_count = [0 for _ in range(singer_count+1)]

for _ in range(pd_count):
    temp = list(map(int, sys.stdin.readline().split()))
    for i in range(1, temp[0]):
        graph[temp[i]].append(temp[i+1])
        line_count[temp[i+1]]+=1

nodes = deque()
for i in range(1, singer_count+1):
    if line_count[i] == 0:
        nodes.append(i)

result = list()
while nodes:
    pop_node = nodes.popleft()
    result.append(pop_node)
    for g in graph[pop_node]:
        line_count[g] -= 1
        if line_count[g] == 0:
            nodes.append(g)

if len(result) == singer_count:
    print(*result, sep='\n')
else:
    print(0)